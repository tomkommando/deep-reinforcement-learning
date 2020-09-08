# We train an agent to map states to action values

import gym

import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time
from dqn_agent import Agent  # import agent.py

# add path to runtime to manage file structure
import sys
from paths import SYSPATH, WEIGHTS_FILE_PATH

sys.path.insert(1, SYSPATH)
# save and load weights
WEIGHTS_PATH = WEIGHTS_FILE_PATH
WEIGHTS_FILE = WEIGHTS_FILE_PATH + "checkpoint_solved1597438700.pth"

# Train or load existing network weights
TRAIN = False

# instantiate environment
env = gym.make('LunarLander-v2')
env.seed(0)
# print('State shape: ', env.observation_space.shape)
# print('Number of actions: ', env.action_space.n)

# instantiate agent
action_size = env.action_space.n  # pick environment n of actions
state_size = env.observation_space.shape[0]  # pick environment state size
agent = Agent(state_size, action_size, seed=0)  # initialise agent

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            # with probability of epsilon, select a random action a, otherwise select argmax action, from action Q table
            action = agent.act(state, eps)
            # take action in emulator and observe next state, reward, and whether is done
            next_state, reward, done, _ = env.step(action)

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            weights_file_name = WEIGHTS_FILE_PATH + 'checkpoint_episode_' + str(i_episode) + '.pth'
            torch.save(agent.qnetwork_local.state_dict(), weights_file_name)
        if np.mean(scores_window) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_window)))
            # save trained model weights with a timestamp
            weights_file_name = WEIGHTS_FILE_PATH + 'checkpoint_solved' + str(int(round(time.time(), 0))) + '.pth'
            torch.save(agent.qnetwork_local.state_dict(), weights_file_name)
            break
    return scores

def play(weights_file):
    # load the weights from file
    agent.qnetwork_local.load_state_dict(torch.load(weights_file))

    for i in range(5):
        state = env.reset()
        for j in range(500):
            action = agent.act(state)
            env.render()
            state, reward, done, _ = env.step(action)
            if done:
                break

# watch trained agent playing or else train model
if TRAIN == False:
    play(WEIGHTS_FILE)

else:
    start_time = time.time()
    scores = dqn()
    print("--- Training took %s seconds ---" % (time.time() - start_time))

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

env.close()
