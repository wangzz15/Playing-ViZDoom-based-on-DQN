import os
import random
import numpy as np
import cv2
import torch
from vizdoom import *

from DQN import DQN, StateMotion


def ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


save_model = True
load_model = False
load_model_dir = os.path.join('weights', 'DQN.pkl')

filepath = "./data.txt"
weights_folder = os.path.join('weights')
ensure_dir_exists(weights_folder)

cfg = {}
cfg['learning_rate'] = 5e-4
cfg['min_epsilon'] = 0.1
cfg['max_epsilon'] = 0.3 if load_model else 1.
cfg['epsilon_decay'] = 10
cfg['print_interval'] = 100
cfg['save_interval'] = 10
cfg['learning_start'] = 5
cfg['max_episodes'] = 5000
cfg['repeat_frame'] = 4
cfg['num_frames'] = 1

FRAME_CHANNEL = 3
RSZ_H = 27
RSZ_W = 36


def preprocess(obs):
    obs = obs.swapaxes(0, 2)
    obs = cv2.resize(obs, (RSZ_H, RSZ_W))
    obs = obs.swapaxes(0, 2)
    return obs


def exploration_rate(i_episode):
    const_eps_epi = 0.1*cfg['max_episodes']
    decay_eps_epi = 0.6*cfg['max_episodes']

    if i_episode < const_eps_epi:
        epsilon = cfg['max_epsilon']
    elif i_episode < decay_eps_epi:
        epsilon = cfg['max_epsilon'] - cfg['min_epsilon']
        epsilon *= i_episode - const_eps_epi
        epsilon /= decay_eps_epi - const_eps_epi
        epsilon = cfg['max_epsilon'] - epsilon
    else:
        epsilon = cfg['min_epsilon']

    return epsilon


def train():
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

    # Define the DQN agent
    model = DQN(frame_shape=(RSZ_H, RSZ_W, FRAME_CHANNEL),
                num_frames=cfg['num_frames'],
                num_actions=num_actions,
                gamma=0.98,
                buffer_size=10000,
                batch_size=32,
                learning_rate=cfg['learning_rate'],
                target_q_update_freq=5,
                double_q=False,
                dueling=False,
                device='cuda:0')

    # tmp_sm = StateMotion(frame_shape=(RSZ_H, RSZ_W, FRAME_CHANNEL),
    #                      num_frames=5)
    # frame_base = os.path.join('state_motions')
    # ensure_dir_exists(frame_base)

    game = DoomGame()
    game.load_config("..\\scenarios\\basic.cfg")
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.init()

    f = open(filepath, 'w')

    if load_model:
        model.q.load_state_dict(torch.load(load_model_dir))
        model.q_target.load_state_dict(model.q.state_dict())

    frame_iter = 0
    for i_episode in range(cfg['learning_start']):
        game.new_episode()
        state_motion = StateMotion(frame_shape=(RSZ_H, RSZ_W, FRAME_CHANNEL),
                                   num_frames=cfg['num_frames'])
        next_state_motion = StateMotion(frame_shape=(RSZ_H, RSZ_W, FRAME_CHANNEL),
                                        num_frames=cfg['num_frames'])

        obs = preprocess(game.get_state().screen_buffer)
        state_motion.update_motion(obs)
        next_state_motion.frame_array = state_motion.frame_array.copy()
        # tmp_sm.update_motion(obs)
        # frame_folder = os.path.join(
        #     frame_base, 'step_{:04d}'.format(frame_iter))
        # ensure_dir_exists(frame_folder)
        # cs = 0
        # for i in range(tmp_sm.num_frames):
        #     ce = cs + tmp_sm.frame_c
        #     tmp_frame = tmp_sm.frame_array[cs:ce, :, :].copy()
        #     tmp_frame = tmp_frame.transpose((1, 2, 0))
        #     cv2.imwrite(os.path.join(
        #         frame_folder, 'frame_{:04d}.png'.format(i)), tmp_frame)
        #     cs = ce
        # frame_iter += 1

        epi_reward = 0.0
        done = False
        while not done:
            action = random.randint(0, num_actions - 1)
            reward = game.make_action(actions[action], cfg['repeat_frame'])

            done = game.is_episode_finished()
            done_mask = 0.0 if done else 1.0

            next_obs = np.zeros((FRAME_CHANNEL, RSZ_H, RSZ_W), dtype=float) if done else preprocess(
                game.get_state().screen_buffer)
            next_state_motion.update_motion(next_obs)

            # tmp_sm.update_motion(next_obs)
            # frame_folder = os.path.join(
            #     frame_base, 'step_{:04d}'.format(frame_iter))
            # ensure_dir_exists(frame_folder)
            # cs = 0
            # for i in range(tmp_sm.num_frames):
            #     ce = cs + tmp_sm.frame_c
            #     tmp_frame = tmp_sm.frame_array[cs:ce, :, :].copy()
            #     tmp_frame = tmp_frame.transpose((1, 2, 0))
            #     cv2.imwrite(os.path.join(
            #         frame_folder, 'frame_{:04d}.png'.format(i)), tmp_frame)
            #     cs = ce
            # frame_iter += 1

            # save the data into the replay buffer
            model.memory.insert((state_motion.frame_array.copy(),
                                 action, 
                                 reward / 100.0,
                                 next_state_motion.frame_array.copy(),
                                 done_mask))

            state_motion.frame_array = next_state_motion.frame_array.copy()

    avg_score = 0.0
    avg_loss = 0.0
    for i_episode in range(cfg['max_episodes']):
        # change the probability of random action according to the training process
        epsilon = exploration_rate(i_episode)

        game.new_episode()
        state_motion = StateMotion(frame_shape=(RSZ_H, RSZ_W, FRAME_CHANNEL),
                                   num_frames=cfg['num_frames'])
        next_state_motion = StateMotion(frame_shape=(RSZ_H, RSZ_W, FRAME_CHANNEL),
                                        num_frames=cfg['num_frames'])

        obs = preprocess(game.get_state().screen_buffer)
        state_motion.update_motion(obs)
        next_state_motion.frame_array = state_motion.frame_array.copy()

        epi_reward = 0.0
        epi_loss = 0.0
        done = False
        while not done:
            action = model.sample_action(state_motion.frame_array, epsilon)
            reward = game.make_action(actions[action], cfg['repeat_frame'])

            done = game.is_episode_finished()
            done_mask = 0.0 if done else 1.0

            next_obs = np.zeros((FRAME_CHANNEL, RSZ_H, RSZ_W), dtype=float) if done else preprocess(
                game.get_state().screen_buffer)
            next_state_motion.update_motion(next_obs)

            # save the data into the replay buffer
            model.memory.insert((state_motion.frame_array.copy(),
                                 action, reward / 100.0,
                                 next_state_motion.frame_array.copy(),
                                 done_mask))

            # record the data
            epi_reward += reward
            epi_loss += model.learn()

            state_motion.frame_array = next_state_motion.frame_array.copy()

        avg_score += epi_reward
        avg_loss += epi_loss

        if i_episode % cfg['print_interval'] == 0:
            if i_episode > 0:
                avg_score /= cfg['print_interval']
                avg_loss /= cfg['print_interval']
                print("episode: {}, avg score: {:.2f}, loss: {:.5f}, buffer size: {}, epsilon:{:.2f}%"
                      .format(i_episode, avg_score, avg_loss,
                              model.memory.size(), epsilon * 100))
                s = str(avg_score) + ' ' + str(avg_loss) + '\n'
                f.write(s)
                avg_score = 0.0
                avg_loss = 0.0

                if save_model:
                    torch.save(model.q.state_dict(),
                               os.path.join(weights_folder,
                                            'params_epi_{:05d}.pkl'.format(i_episode)))
            else:
                avg_score = 0.0
                avg_loss = 0.0
    f.close()
    print('The training process is end.')


if __name__ == "__main__":
    train()
