import numpy as np
import argparse
from roomba_env import RoombaEnv
from neural_network import create_Q_network, test_model

def parse_args():
    parser = argparse.ArgumentParser(description="Test the Lunar Lander model.")
    parser.add_argument('--gamma', type=float, default=0.99, required=False, help="Gamma value")
    parser.add_argument('--checkpoint', type=str, help="Saved model")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    env = RoombaEnv()
    n_actions = env.action_space.n
    model = create_Q_network(env.observation_space.shape, n_actions)
    model.load_weights(args.checkpoint)
    env._max_episode_steps = 1000

    # test_model(model, env)
    for i_episode in range(10):
        observation = env.reset()
        total_reward = 0
        done = False
        t=0
        while not done:
            env.render()
            #print("State ", observation)
            q_values = model.predict(np.array([observation]))
            # if np.random.rand() < 0.1:
            #     action = np.random.randint(n_actions)
            # else:
            #     action = np.argmax(q_values)
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
