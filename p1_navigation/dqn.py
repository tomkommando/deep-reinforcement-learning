import torch
import numpy as np
from collections import deque
import time


# if you have pre loaded weights in place. Set eps_start=0
def dqn(env, agent, WEIGHTS_PATH, brain_name, n_episodes=2000, eps_start=1, eps_end=0.01, eps_decay=0.993):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # track episode scores
    yellow_bananas = []  # track episode yellow bananas
    blue_bananas = []  # track episode blue bananas
    steps = []  # track episode steps
    epsilons = []  # track episode epsilons
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    for i_episode in range(1, n_episodes + 1):

        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0  # initialize the score
        n_steps = 0  # initialize steps
        n_yellow_bananas = 0
        n_blue_bananas = 0

        while True:
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            score += reward  # update the score
            n_steps += 1
            if reward == -1:
                n_blue_bananas += 1
            if reward == 1:
                n_yellow_bananas += 1
            agent.step(state, action, reward, next_state, done)
            state = next_state  # roll over the state to next time step
            if done:  # exit loop if episode finished
                break

        # append performance metrics to lists
        scores_window.append(score)
        scores.append(score)
        steps.append(n_steps)
        yellow_bananas.append(n_yellow_bananas)
        blue_bananas.append(n_blue_bananas)
        epsilons.append(eps)

        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        # track training episodes and save weight file checkpoints
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.4f}'.format(i_episode, np.mean(scores_window), eps))
            weights_file_name = WEIGHTS_PATH + 'checkpoint_episode_' + str(i_episode) + '.pth'
            torch.save(agent.qnetwork_local.state_dict(), weights_file_name)
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_window)))
            # save trained model weights with a timestamp
            weights_file_name = WEIGHTS_PATH + 'checkpoint_solved' + str(int(round(time.time(), 0))) + '.pth'
            torch.save(agent.qnetwork_local.state_dict(), weights_file_name)
            break

    return scores, steps, yellow_bananas, blue_bananas, epsilons
