import os
import time
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from vizdoom import *

from DQN import DQNetwork, StateMotion

load_model_dir = os.path.join('weights', 'DQN.pkl')
visible = False
duel = False

cfg = {}
cfg['test_episodes'] = 200
cfg['repeat_frame'] = 4
cfg['num_frames'] = 1

FRAME_CHANNEL = 3
RSZ_H = 27
RSZ_W = 36

DEVICE = 'cuda:0'


def preprocess(obs):
    obs = obs.swapaxes(0, 2)
    obs = cv2.resize(obs, (RSZ_H, RSZ_W))
    obs = obs.swapaxes(0, 2)
    return obs


def test():
    # actions = [[False, False, False],
    #            [False, False, True],
    #            [False, True, False],
    #            [False, True, True],
    #            [True, False, False],
    #            [True, False, True],
    #            [True, True, False],
    #            [True, True, True]]
    actions = [[True, False, False],
               [False, False, True],
               [False, True, False]]
    num_actions = len(actions)
    model = DQNetwork(w=RSZ_W, h=RSZ_H,
                      c=FRAME_CHANNEL * cfg['num_frames'], dueling=duel).to(DEVICE)
    model.load_state_dict(torch.load(load_model_dir))

    score = 0.0

    game = DoomGame()
    game.load_config("..\\scenarios\\basic.cfg")
    game.set_window_visible(visible)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    with torch.no_grad():
        avg_score = 0.0
        min_score = float('inf')
        max_score = 0.0
        for i_episode in range(cfg['test_episodes']):
            game.new_episode()
            state_motion = StateMotion(frame_shape=(RSZ_H, RSZ_W, FRAME_CHANNEL),
                                       num_frames=cfg['num_frames'])

            obs = preprocess(game.get_state().screen_buffer)
            state_motion.update_motion(obs)

            epi_reward = 0.0
            done = False
            while not done:

                torch_state = torch.from_numpy(
                    state_motion.frame_array).unsqueeze(0).float().to(DEVICE)
                action = np.argmax(model.forward(
                    torch_state).to('cpu').numpy())
                reward = game.make_action(actions[action], cfg['repeat_frame'])

                done = game.is_episode_finished()
                next_obs = np.zeros((FRAME_CHANNEL, RSZ_H, RSZ_W), dtype=float) if done else preprocess(
                    game.get_state().screen_buffer)
                state_motion.update_motion(next_obs)

                epi_reward += reward

            avg_score += epi_reward
            min_score = min(min_score, epi_reward)
            max_score = max(max_score, epi_reward)
            print('episode: {}, score: {:.2f}'.format(i_episode, epi_reward))
            if visible:
                time.sleep(1)

        avg_score /= cfg['test_episodes']
        print('{} episodes tested, average score: {:.2f}'.format(
            cfg['test_episodes'], avg_score))
        print('min score', min_score)
        print('max score', max_score)



if __name__ == "__main__":
    test()
