
from time import time
import numpy as np
import copy
import argparse
import sys
from roomba_env import RoombaEnv
from neural_network import train, create_Q_network, test_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Lunar Lander model.")
    parser.add_argument('--gamma', type=float, default=0.99, required=False, help="Gamma value")
    parser.add_argument('--episodes', type=int, default=1500, required=False)
    parser.add_argument('--C', type=int, default=50, required=False)
    parser.add_argument('--learning_rate', type=float, default=0.00025)
    parser.add_argument('--initial_epsilon', type=float, default=1.0)
    parser.add_argument('--final_epsilon', type=float, default=0.1)
    parser.add_argument('--replay_memory_size', type=int, default=10000)
    return parser.parse_args()

if __name__ == '__main__':
    #env = gym.make('CartPole-v0')
    env = RoombaEnv(max_episode_steps=1000)
    n_actions = env.action_space.n
    args = parse_args()
    sys.stdout=open('gamma_{}_episodes_{}_C_{}_replay_{}'.format(args.gamma, args.episodes, args.C, args.replay_memory_size), 'w')
    output_file = './checkpoints/gamma_{}_episodes_{}_C_{}_replay_{}'.format(args.gamma, args.episodes, args.C, args.replay_memory_size)
    model = train(env, n_actions, args)
    model.save_weights(output_file)
    model.load_weights(output_file)
    env._max_episode_steps = 1000

    test_model(model, env)
    for i_episode in range(10):
        observation = env.reset()
        total_reward = 0
        done = False
        t=0
        while not done:
            env.render()
            #print("State ", observation)
            q_values = model.predict(np.array([observation]))
            if np.random.rand() < 0.1:
                action = np.random.randint(n_actions)
            else:
                action = np.argmax(q_values)
            #print("Action; ", action)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            t += 1
            if done:
                #print("Episode finished after {} timesteps".format(t+1))
                #print("Reward=", total_reward)
                break
    env.close()
