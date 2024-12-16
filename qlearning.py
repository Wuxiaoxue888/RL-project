from datetime import datetime
from math import cos, pi

import tensorflow as tf
import numpy as np
np.set_printoptions(precision=3)

np.random.seed(42)
tf.random.set_seed(42)

from mahjong_env import *

env = gym.make("Mahjong-v2")

input_shape = [108]
n_outputs = 27

model = tf.keras.Sequential([
    tf.keras.Input(input_shape),
    tf.keras.layers.Dense(64, activation="elu"),
    tf.keras.layers.Dense(64, activation="elu"),
    tf.keras.layers.Dense(n_outputs)
])

def epsilon_greedy_policy(observation, epsilon):
    observation = np.concatenate([observation[x] for x in sorted(observation)])
    if np.random.rand() < epsilon:
        action = np.random.choice(np.nonzero(observation[:27])[0])
    else:
        action_q_values = model.predict(observation[np.newaxis], verbose=0)[0]
        action_q_values[(observation[:27]==0)] = -np.inf # mask invalid actions
        action = np.argmax(action_q_values)
    return action

class CircularReplayBuffer:
    def __init__(self, max_size):
        self.buffer = np.empty(max_size, dtype=object)
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj):
        observation, action, reward, next_observation, done, truncated = obj
        observation = np.concatenate([observation[x] for x in sorted(observation)])
        next_observation = np.concatenate([next_observation[x] for x in sorted(next_observation)])

        self.buffer[self.index] = (observation, action, reward, next_observation, done, truncated)
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        indices = np.random.randint(self.size, size=batch_size)
        return self.buffer[indices]

class PrioritizedReplayBuffer(CircularReplayBuffer):
    def __init__(self, max_size):
        super(PrioritizedReplayBuffer, self).__init__(max_size)

    def sample(self, batch_size):
        priorities = [abs(experience[2]) for experience in self.buffer]
        probabilities = priorities / np.sum(priorities)
        sample_indices = np.random.choice(
            range(len(self.buffer)), size=batch_size, p=probabilities)
        return [self.buffer[i] for i in sample_indices]

REPLAY_BUFFER_SIZE = 2_000
replay_buffer = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE)

def sample_experiences(batch_size):
    batch = replay_buffer.sample(batch_size)
    return [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(6)
    ]

def play_one_step(env, observation, epsilon):
    action = epsilon_greedy_policy(observation, epsilon)
    next_observation, reward, done, truncated, info = env.step(action)
    replay_buffer.append((observation, action, reward, next_observation, done, truncated))
    return next_observation, reward, done, truncated, info

env.reset(seed=42)
np.random.seed(42)
tf.random.set_seed(42)
rewards = []
best_score = -10_000

batch_size = 32
discount_factor = 0.95
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.mse

def training_step(batch_size):
    observations, actions, rewards, next_observations, dones, truncateds = sample_experiences(batch_size)
    next_q_values = model.predict(next_observations, verbose=0)
    max_next_q_values = next_q_values.max(axis=1)
    next_observation_exists = 1.0 - (dones | truncateds)
    target_Q_values = rewards + next_observation_exists * discount_factor * max_next_q_values
    target_Q_values = target_Q_values.reshape(-1, 1)

    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(observations)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

EPSILON_INITIAL = 0.50
EPSILON_MIN = 0.05
DECAY_EPISODES = 3000
def epsilon_decay(ep):
    return EPSILON_MIN + 0.5 * (EPSILON_INITIAL - EPSILON_MIN) * (
            1 + cos(pi * ep / DECAY_EPISODES))

def test_network():
    observation, _ = env.reset()
    episode_reward = 0
    while True:
        observation, reward, done, truncated, info = play_one_step(env, observation, 0)
        episode_reward += reward
        if done or truncated:
            break

    return episode_reward

from reward_writer import RewardWriter

episode_reward_writer = RewardWriter("mahjong_training.csv", frequency=100)
testing_reward_writer = RewardWriter("mahjong_testing.csv")

episode = 0
test_every_n = 300
while True:
    observation, _ = env.reset()
    episode_reward = 0
    step = 0
    while True:
        epsilon = epsilon_decay(episode)
        observation, reward, done, truncated, info = play_one_step(env, observation, epsilon)
        episode_reward += reward
        if done or truncated:
            break
        step += 1

    # Episode ended
    print(f"({datetime.now()}) Episode: {episode}, Steps: {step + 1}, Reward: {episode_reward:.2f}, eps: {epsilon:.3f}")

    episode_reward_writer.write_reward(episode, episode_reward)

    # Test current network
    if episode % test_every_n == 0:
        testing_reward_writer.write_reward(episode, test_network())
        model.save(f"model-{episode}.keras")

    if episode != 0 and episode % 100 == 0 and replay_buffer.size == REPLAY_BUFFER_SIZE:
            print(f"({datetime.now()}) Training the model...")
            training_step(batch_size)

    episode += 1
