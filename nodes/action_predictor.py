#!/usr/bin/env python3

"""
This node performs inference for a previously trained diffusion model. It 
receives observation data from nodes in an additional package, and sends the 
model output action sequences to nodes for execution on the Franka robot.

PUBLISHERS:
  + /predicted_action (Float32MultiArray) - The puptput actions to be executed on the franka
  + /action_horizon (Float32MultiArray) - The full action horizon output from the  model
SUBSCRIBERS:
  + /scene_image_obs (Image) - The scene image observation
  + /ee_image_obs (Image) - The end effector image observation
  + /current_ee_pose (Pose) - The end effector position observation
SERVICE:
  + /start_diffusion (Empty) - Triggers the next diffusion inference after robot execution is complete.
  + /enable_diffusion (Empty) - Triggers the starting of the diffusion inference loop.
  + /disable_diffusion (Empty) - Triggers the stopping of the diffusion inference loop.
"""

import torch
import dill
import hydra
import numpy as np
import threading
from enum import auto as enum_auto
import time

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from std_srvs.srv import Empty

from cv_bridge import CvBridge


class ActionPredictor(Node):
    """ROS node for predicting franka actions using Diffusion Policy."""
    def __init__(self):
        super().__init__('action_predictor')

        # frequency parameter
        self.declare_parameter('frequency', 30.0, ParameterDescriptor(description='Frequency (hz) of the timer callback'))
        self.timer_freqency = self.get_parameter('frequency').get_parameter_value().double_value

        # checkpoint path parameter
        self.declare_parameter(
            'checkpoint_path',
            '',
            ParameterDescriptor(description='Checkpoint file (.ckpt) that contains model weights and config info')
        )
        self.checkpoint_path = self.get_parameter('checkpoint_path').get_parameter_value().string_value
        if self.checkpoint_path == '':
            raise Exception('No checkpoint provided for loading model')
        
        # Subscribers:
        self.scene_img_sub = self.create_subscription(Image, 'scene_image_obs', self.scene_image_callback, 10)
        self.current_ee_sub = self.create_subscription(Pose, 'current_ee_pose', self.current_ee_callback, 10)
        self.ee_img_sub = self.create_subscription(Image, 'ee_image_obs', self.ee_image_callback, 10)

        # Services
        self.enable_diffusion_srv = self.create_service(Empty, 'enable_diffusion', self.enable_diffusion_srv_callback)
        self.disable_diffusion_srv = self.create_service(Empty, 'disable_diffusion', self.disable_diffusion_srv_callback)
        self.start_diffusion_srv = self.create_service(Empty, 'start_diffusion', self.start_diffusion_srv_callback)
        self.enable_diffusion = False
        self.enable_inference = False
        self.enable_action = False

        # Publishers
        self.predicted_action_pub = self.create_publisher(Float32MultiArray, 'predicted_action', 10)
        self.action_horizon_pub = self.create_publisher(Float32MultiArray, 'action_horizon', 10)

        # Load checkpoint
        self.payload = torch.load(open(self.checkpoint_path, 'rb'), pickle_module=dill)
        self.cfg = self.payload['cfg']

        # Load workspace
        workspace_cls = hydra.utils.get_class(self.cfg._target_)
        self.workspace: BaseWorkspace = workspace_cls(self.cfg)
        self.workspace.load_payload(self.payload, exclude_keys=None, include_keys=None)

        # Load model
        self.policy: BaseImagePolicy = self.workspace.model
        if self.cfg.training.use_ema:
            self.policy = self.workspace.ema_model

        # number of inference steps parameter
        self.declare_parameter('num_inference_steps', 100, ParameterDescriptor(description='Number of Inference Steps'))
        self.num_inference_steps = self.get_parameter('num_inference_steps').get_parameter_value().integer_value

        self.get_logger().info(f'Policy Inference Steps Before = {self.policy.num_inference_steps}')
        self.policy.num_inference_steps = self.num_inference_steps
        self.get_logger().info(f'Policy Inference Steps After = {self.policy.num_inference_steps}')

        # Enable GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.eval().to(self.device)

        # Create timer
        self.timer = self.create_timer((1.0/self.timer_freqency), self.timer_callback)

        self.bridge = CvBridge()

        self.get_logger().info(f'Policy # Actions Before = {self.policy.n_action_steps}')
        self.declare_parameter(
            'num_actions_taken',
            self.policy.n_action_steps,
            ParameterDescriptor(
                description='Number of actions taken based on an inference before a new inference is used'
            )
        )
        self.num_actions_taken = self.get_parameter('num_actions_taken').get_parameter_value().integer_value
        if self.num_actions_taken > self.policy.n_action_steps:
            self.num_actions_taken = self.policy.n_action_steps
        if self.num_actions_taken < self.policy.n_action_steps:
            self.policy.n_action_steps = self.num_actions_taken
        self.get_logger().info(f'Policy # Actions After = {self.policy.n_action_steps}')

        # track received messages
        self.reset_obs_received()
        self.at_least_one_of_each_obs_received = False
        self.latest_message = [None, None, None]
        self.obs_recieved = np.array([False, False, False])
        self.observation_queue = []
        self.obs_data_mutex = threading.Lock()

        # track inference
        self.inference_thread = threading.Thread()

        # track actions
        self.action_data_mutex = threading.Lock()
        self.action_array = []
        self.send_full_action = True

        # model info
        self.action_start = self.policy.n_obs_steps - 1
        self.action_end = self.action_start + self.policy.n_action_steps
        self.get_logger().info(f'Horizon = {self.policy.horizon}')
        self.get_logger().info(f'# Observations = {self.policy.n_obs_steps}')
        self.get_logger().info(f'# Action Steps = {self.policy.n_action_steps}')
        self.get_logger().info(f'Start Action Index = {self.action_start}')
        self.get_logger().info(f'End Action Index = {self.action_end}')

    def reset_obs_received(self):
        """Resets the observations."""
        self.obs_recieved = np.array([False, False, False])

    def scene_image_callback(self, msg):
        """Appends the most recent scene image received to the observation."""
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        tp_img = np.moveaxis(img, -1, 0)
        self.latest_message[0] = np.array(tp_img).astype(np.float32)/255.0
        self.obs_recieved[0] = True
    
    def ee_image_callback(self, msg):
        """Appends the most recent end effector image received to the observation."""
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8', )
        tp_img = np.moveaxis(img, -1, 0)
        self.latest_message[1] = np.array(tp_img).astype(np.float32)/255.0
        self.obs_recieved[1] = True

    def current_ee_callback(self, msg):
        """Appends the most recent end effector position received to the observation."""
        self.get_logger().info('Received Current EE Pose', once=True)
        self.latest_message[2] = np.array([msg.position.x, msg.position.y]).astype(np.float32)
        self.obs_recieved[2] = True
    
    def enable_diffusion_srv_callback(self, request, response):
        """Triggers the starting of the inference loop."""
        self.get_logger().info('Diffusion Enabled')
        self.enable_inference = True
        self.enable_diffusion = True
        return response
    
    def disable_diffusion_srv_callback(self, request, response):
        """Triggers the stopping of the inference loop."""
        self.get_logger().info('Diffusion Disabled')
        self.enable_inference = False
        self.enable_action = False
        self.enable_diffusion = False
        return response
    
    def start_diffusion_srv_callback(self, request, response):
        """Triggers the starting of an inference"""
        self.get_logger().info('Diffusion Called')
        self.enable_inference = True
        self.action_array = []
        self.send_full_action = True
        return response
        
    def run_inference(self):
        """Runs the inference process"""

        self.get_logger().info('Starting Inference ...')
        start = time.time()

        with self.obs_data_mutex:

            scene_imgs = []
            ee_imgs = []
            agts = []
            for ii in np.arange(len(self.observation_queue)):
                scene_imgs.append(self.observation_queue[ii][0])
                ee_imgs.append(self.observation_queue[ii][1])
                agts.append(self.observation_queue[ii][2])

            scene_images = np.array(scene_imgs)
            ee_images = np.array(ee_imgs)
            agent_pos = np.array(agts)

        obs_data_tensor = dict_apply(
            {'scene_img': scene_images, 'ee_img': ee_images, 'agent_pos': agent_pos},
            lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            result = self.policy.predict_action(obs_data_tensor)

        with self.action_data_mutex:
            self.action_array = result['action'].squeeze(0).to('cpu').numpy()
            self.action_pred = result['action_pred'].squeeze(0).to('cpu').numpy()
            self.send_full_action = True

        self.get_logger().info('Inference Complete.')
        self.get_logger().info(f'Time: {time.time() - start} sec')
        self.enable_inference = False
        self.enable_action = True

    def timer_callback(self):

        # check if observations for both image and pose exisit
        if np.any(~self.obs_recieved):
            if not self.at_least_one_of_each_obs_received:
                return
        self.at_least_one_of_each_obs_received = True
        self.reset_obs_received()

        # check there are enough observations to run the model
        with self.obs_data_mutex:
            if len(self.observation_queue) < self.cfg.n_obs_steps:
                # add observation if there aren't enough to make an inference yet
                self.observation_queue.append(self.latest_message)
                # do not continue unil there are enough for the model
                return
            else:
                self.observation_queue = self.observation_queue[1:] + [self.latest_message]

        if not self.enable_diffusion:
            return
        
        # check if begining inference
        if self.enable_inference:
            # only start new inference if the last inference has completed
            if not self.inference_thread.is_alive():
                self.inference_thread = threading.Thread(target=self.run_inference)
                self.inference_thread.start()
        
        # complete actions
        with self.action_data_mutex:

            if self.enable_action:

                if self.send_full_action:
                    # crop action array
                    self.get_logger().info(f'Number of actions being sent = {self.action_array.shape[0]}')
                    self.get_logger().info(f'Actions being sent = {self.action_array}')
                    # construct action array message
                    actions_msg = Float32MultiArray()
                    actions_msg.data = self.action_array.flatten().tolist()
                    # add dimensions of flattened array
                    actions_msg.layout.dim.append(MultiArrayDimension())
                    actions_msg.layout.dim[0].label = "rows"
                    actions_msg.layout.dim[0].size = self.action_array.shape[0]
                    actions_msg.layout.dim[0].stride = self.action_array.shape[0] * self.action_array.shape[1]
                    actions_msg.layout.dim.append(MultiArrayDimension())
                    actions_msg.layout.dim[1].label = "columns"
                    actions_msg.layout.dim[1].size = self.action_array.shape[1]
                    actions_msg.layout.dim[1].stride = self.action_array.shape[1]

                    self.predicted_action_pub.publish(actions_msg)
                    self.send_full_action = False

                    # make horizon message
                    horizon_msg = Float32MultiArray()
                    horizon_msg.data = self.action_pred.flatten().tolist()
                    # add dimensions of flattened array
                    horizon_msg.layout.dim.append(MultiArrayDimension())
                    horizon_msg.layout.dim[0].label = "rows"
                    horizon_msg.layout.dim[0].size = self.action_pred.shape[0]
                    horizon_msg.layout.dim[0].stride = self.action_pred.shape[0] * self.action_pred.shape[1]
                    horizon_msg.layout.dim.append(MultiArrayDimension())
                    horizon_msg.layout.dim[1].label = "columns"
                    horizon_msg.layout.dim[1].size = self.action_pred.shape[1]
                    horizon_msg.layout.dim[1].stride = self.action_pred.shape[1]

                    self.action_horizon_pub.publish(horizon_msg)
                
                    self.get_logger().info('Sending new action array')
                    self.enable_action = False
    

def main(args=None):
    rclpy.init(args=args)
    node = ActionPredictor()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()