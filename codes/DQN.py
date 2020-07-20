import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from memory_buffers import ReplayBuffer
from nn_models import DQNetwork


class StateMotion():
    def __init__(self, frame_shape, num_frames, device='cuda:0'):
        self.num_frames = num_frames
        self.frame_h = frame_shape[0]
        self.frame_w = frame_shape[1]
        self.frame_c = frame_shape[2]
        self.frame_array = None

    def update_motion(self, obs):
        # game.new_episode()
        if self.frame_array is None:
            self.frame_array = np.zeros((self.num_frames * self.frame_c,
                                         self.frame_h,
                                         self.frame_w), dtype=np.uint8)
            cs = 0
            for i in range(self.num_frames):
                ce = cs + self.frame_c
                self.frame_array[cs:ce, :, :] = obs.copy()
                cs = ce

        else:
            dest_cs = 0
            for i in range(1, self.num_frames):
                dest_ce = dest_cs + self.frame_c
                src_cs = dest_ce
                src_ce = src_cs + self.frame_c
                self.frame_array[dest_cs:dest_ce, :,
                                 :] = self.frame_array[src_cs:src_ce, :, :].copy()
                dest_cs = dest_ce

            self.frame_array[-self.frame_c:, :, :] = obs.copy()


class DQN():
    def __init__(self,
                 frame_shape,
                 num_frames,
                 num_actions,
                 gamma,
                 buffer_size,
                 batch_size,
                 learning_rate,
                 target_q_update_freq,
                 double_q=False,
                 dueling=False,
                 device='cuda:0'):
        """
        The DQN agent.
        args:
            frame_shape: The shape of the screen of a frame.
                type: tuple (height, width, channel)
            num_frames: The number of frames to construct a motion for a state.
                type: int
            num_actions: The number of the actions
                type: int
            gamma: The discount factor gamma.
                type: float
            buffer_size: The size of the Replay Buffer
                type: int
            batch_size: The size of the mini batch for every updating process.
                type: int
            learning_rate: The learning rate for the q network.
                type: float
            target_q_update_freq: The frequency for updating the target q network.
                type: int
            double_q: If using double Q learning.
                type: boolean
            device: The device to train the networks. ('cpu' or 'cuda')
                type: str
        """
        self.frame_shape = frame_shape
        self.num_frames = num_frames
        self.num_actions = num_actions
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_q_update_freq = target_q_update_freq
        self.double_q = double_q
        self.device = device
        self.dueling = dueling

        self.num_params_update = 0

        self.create_nets()

        self.memory = ReplayBuffer(self.buffer_size)

        self.optimizer = optim.Adam(self.q.parameters(), lr=self.learning_rate)

    def create_nets(self):
        """
        Initialize the q network and target q network. Copy the parameters from q network to target q network.
        """
        self.q = DQNetwork(w=self.frame_shape[1], h=self.frame_shape[0],
                           c=self.frame_shape[2] * self.num_frames, dueling=self.dueling).to(self.device)
        self.q_target = DQNetwork(w=self.frame_shape[1], h=self.frame_shape[0],
                                  c=self.frame_shape[2] * self.num_frames, dueling=self.dueling).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

    def sample_action(self, state_motion, epsilon=0.0):
        """
        Sample the action according to epsilon-greedy policy.
        args:
            obs: observation from the environment.
                type: np.ndarray(dtype=np.float)
                shape: [num_inputs]
            epsilon: The probability to take a random action. Default is 0.0
                type: float
        """
        state_motion = torch.from_numpy(state_motion).unsqueeze(0).float()
        q_out = self.q(state_motion.to(self.device))

        # ramdom sample the acition via epsilon-greedy.
        random_digit = random.random()
        action = q_out.argmax().item() if random_digit >= epsilon else random.randint(
            0, self.num_actions - 1)
        return action

    def calc_loss(self):
        """
        Sample a mini-batch from the replay buffer and then calculate the loss function.
        """
        state_batch, action_batch, reward_batch, next_state_batch, done_mask_batch = self.memory.sample(
            self.batch_size)

        # transfer the data to the target device
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_mask_batch = done_mask_batch.to(self.device)

        
        q_out = self.q(state_batch)
        q_values = q_out.gather(1, action_batch)

        if self.double_q:
            next_s_max_indice = self.q(next_state_batch).detach().max(dim=1)[
                1].unsqueeze(dim=1)
            next_s_target_q = self.q_target(next_state_batch)
            next_s_max_q = next_s_target_q.gather(1, next_s_max_indice)
        else:
            next_s_max_q = self.q_target(next_state_batch).detach().max(dim=1)[
                0].unsqueeze(dim=1)

        expected_q_values = reward_batch + self.gamma * next_s_max_q * done_mask_batch
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        return loss

    def learn(self):
        loss = self.calc_loss()

        # updata the q network using optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.num_params_update += 1

        # update the target q network
        if self.num_params_update % self.target_q_update_freq == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return loss
