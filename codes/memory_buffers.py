import random
import torch
import collections


class ReplayBuffer():
    def __init__(self, buffer_size):
        """
        The Replay Buffer for DQN.
        args:
            buffer_size: The max size of the buffer.
                type: int
        """
        self._buffer = collections.deque(maxlen=buffer_size)

    def insert(self, transition):
        """
        Insert the transition data into the buffer.
        args:
            transition: The tuple of the transition - (state, action, reward, next_state, done_mask)
                type: tuple
        """
        self._buffer.append(transition)

    def size(self):
        return len(self._buffer)

    def _canSample(self, batch_size):
        """
        Ensure the number of samples in buffer is more than batch size.
        args:
            batch_size: The size of the mini batch.
                type: int
        """
        return self.size() >= batch_size

    def sample(self, batch_size):
        """
        Sample the mini-batch and return the tensor.
        args:
            batch_size: The size of the mini batch.
                type: int
        """
        assert self._canSample(batch_size)

        mini_batch = random.sample(self._buffer, batch_size)
        state_lst, action_lst, reward_lst, next_state_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, next_s, done_mask = transition
            state_lst.append(s)
            action_lst.append([a])
            reward_lst.append([r])
            next_state_lst.append(next_s)
            done_mask_lst.append([done_mask])

        return torch.tensor(state_lst, dtype=torch.float), \
            torch.tensor(action_lst), \
            torch.tensor(reward_lst), \
            torch.tensor(next_state_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst)
