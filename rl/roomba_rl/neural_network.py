from time import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
import gym
import numpy as np
import copy
import argparse
import sys

def create_Q_network(input_size, output_size):
    inputs = Input(shape=input_size)
    layer1 = Dense(128, activation='relu')(inputs)
    layer2 = Dense(128, activation='relu')(layer1)
    outputs = Dense(output_size)(layer2)
    return keras.models.Model(inputs=inputs, outputs=outputs)


def train(env, n_actions, args):
    # Record tensorboard code
    #writer = tf.summary.FileWriter("/tmp/lunarlander")
    #tf.compat.v1.disable_eager_execution()

    reward_memory = []
    state_memory = []
    state_next_memory = []
    action_memory = []
    terminal_memory = []
    t = 0
    episodes = args.episodes
    update_period = 4
    batch_size = 32
    gamma = args.gamma
    initial_epsilon = 1.0
    final_epsilon = 0.1
    C = 50 # Smaller because the number of frames is smaller
    final_exploration_count = 1000
    replay_memory_size = args.replay_memory_size # 100000
    replay_start_size = 100

    alpha=args.learning_rate
    gradient_momentum=0.9
    squared_gradient_momentum=0.9
    rms_epsilon=0.01
    #optimizer = keras.optimizers.RMSprop(learning_rate=alpha, rho=squared_gradient_momentum, momentum=gradient_momentum, epsilon=rms_epsilon)
    optimizer = keras.optimizers.Adam(learning_rate=alpha)
    loss = keras.losses.Huber()
    model = create_Q_network(env.observation_space.shape, n_actions)
    model_target = create_Q_network(env.observation_space.shape, n_actions)
    model_target.set_weights(model.get_weights())
    for i in range(episodes):
        state = env.reset()
        terminal_state = False
        total_reward=0
        print("Episode: ", i)
        print("=============================")
        print(" timesteps: ", t)
        while not terminal_state:
            t += 1
            epsilon = initial_epsilon - (initial_epsilon - final_epsilon) * (t-replay_start_size) / final_exploration_count
            if t < replay_start_size or np.random.rand() < epsilon:
                action = np.random.randint(n_actions)
            else:
                result = model.predict(np.array([state,]))
                action = np.argmax(result[0])


            # Take step
            next_state, reward, terminal_state, _ = env.step(action)
            total_reward += reward
            if terminal_state:
                print("Reward: ", total_reward)

            # Update memory
            if len(state_memory) >= replay_memory_size:
                state_memory[t%replay_memory_size] = state
                state_next_memory[t%replay_memory_size] = next_state
                reward_memory[t%replay_memory_size] = reward
                action_memory[t%replay_memory_size] = action
                terminal_memory[t%replay_memory_size] = terminal_state
            else:
                state_memory.append(state)
                state_next_memory.append(next_state)
                reward_memory.append(reward)
                action_memory.append(action)
                terminal_memory.append(terminal_state)

            if t%update_period == 0 and len(state_memory) >= replay_start_size:
                #j_update = np.arange(batch_size)
                j_update = np.random.choice(np.arange(len(state_memory)), size=batch_size, replace=False)
                state_j = np.array([state_memory[j] for j in j_update])
                state_next_j = np.array([state_next_memory[j] for j in j_update])
                action_j = [action_memory[j] for j in j_update]
                reward_j = np.array([reward_memory[j] for j in j_update])
                terminal_j = np.array([terminal_memory[j] for j in j_update])
                #print("States: ")
                #print(state_next_j)

                # Use terminal flag as a mask
                # FIXIT: use target rather than the predicted
                # print("Terminal: ", terminal_j)
                q_actual = reward_j + gamma * (1 - terminal_j) * tf.reduce_max(model_target.predict(state_next_j), axis=1)

                action_one_hot = tf.one_hot(action_j, n_actions)
                with tf.GradientTape() as tape:
                    prediction = model(state_j)
                    q_estimated = tf.reduce_sum(tf.multiply(prediction, action_one_hot), axis=1)
                    loss_value = loss(q_actual, q_estimated)
                    #if t%C == 0:
                        #print("Q actual: ", q_actual)
                        #print("Q estimated: ", np.average(q_estimated))
                        #print("Loss: ", np.average(loss_value.numpy()))
                        #print("Loss min/max: ", np.min(loss_value.numpy()), np.max(loss_value.numpy()))
                #with tf.name_scope("performance"):
                #    tf_loss_summary = tf.summary.scalar('loss', loss_value)
                #writer.add_summary(tf_loss_summary, t)
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if t%C == 0:
                model_target.set_weights(model.get_weights())
            state = list(next_state)
    return model

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

# Determine the average reward
def test_model(model, env):
    total_total_reward = 0
    n_actions = env.action_space.n
    for i_episode in range(100):
        observation = env.reset()
        total_reward = 0
        done = False
        t=0
        while not done:
            q_values = model.predict(np.array([observation]))
            if np.random.rand() < 0.1:
                action = np.random.randint(n_actions)
            else:
                action = np.argmax(q_values)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            t += 1
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                print("Reward=", total_reward)
                total_total_reward += total_reward
                break
    env.close()
    print("Average reward: ", total_total_reward / 100)

if __name__ == '__main__':
    #env = gym.make('CartPole-v0')
    env = gym.make('LunarLander-v2')
    env._max_episode_steps = 1000
    n_actions = env.action_space.n
    model = create_Q_network(env.observation_space.shape, n_actions)
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
