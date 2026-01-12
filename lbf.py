import gymnasium as gym
import lbforaging
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools

from collections import defaultdict
from typing import List, DefaultDict
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim
from gymnasium.wrappers import RecordVideo
import csv
import os

import imageio
import pyglet
from PIL import Image
import time


# Observation utilities
def obs_to_key(obs):
    """
    Converts a numpy observation into a hashable tuple key.
    """
    return tuple(obs.astype(int))

def joint_obs_to_key(obss):
    """
    Converts a list of numpy observations into a single hashable joint key.
    """
    return tuple(obs_to_key(o) for o in obss)

# IQL Agent
class IQL:
    """
    Agent using the Independent Q-Learning algorithm.
    Each agent maintains its own Q-table based only on its local observation.
    """
    def __init__(self, num_agents, action_spaces, gamma, learning_rate=0.5, epsilon=1.0):
        """
        Constructor of IQL

        :param num_agents (int): number of agents
        :param action_spaces (List[Space]): action spaces of the environment for each agent
        :param gamma (float): discount factor
        :param learning_rate (float): learning rate
        :param epsilon (float): exploration rate
        """
        self.num_agents = num_agents
        self.n_acts = [flatdim(a) for a in action_spaces]
        self.gamma = gamma
        self.lr = learning_rate
        self.epsilon = epsilon
        # Optimistic inicialization
        self.q_tables = [defaultdict(lambda: 0.01) for _ in range(num_agents)]

    def act(self, obss):
        """
        Selects actions for all agents using Independent Epsilon-Greedy policies.

        :param obss (List): list of local observations
        :return (List[int]): selected actions
        """
        actions = []
        for i in range(self.num_agents):
            obs = obs_to_key(obss[i])

            if random.random() < self.epsilon:
                action = random.randrange(self.n_acts[i])
            else:
                q_vals = [self.q_tables[i][(obs, a)] for a in range(self.n_acts[i])]
                # a = int(np.argmax(q_vals))
                max_q = max(q_vals)
                best_actions = [a for a, q in enumerate(q_vals) if q == max_q]
                action = random.choice(best_actions) # if there are multiple best_actions, choose randomly

            actions.append(action)

        return actions

    def learn(self, obss, actions, rewards, n_obss, done):
        """
        Updates each agent's Q-table independently.

        :param obss: current observations
        :param actions: actions taken
        :param rewards: rewards received
        :param n_obss: next observations
        :param done: terminal flag
        """
        for i in range(self.num_agents):
            obs = obs_to_key(obss[i])
            next_obs = obs_to_key(n_obss[i])
            a = actions[i]
            r = rewards[i]

            q_sa = self.q_tables[i][(obs, a)]
            max_next = 0.0 if done else max(
                self.q_tables[i][(next_obs, ap)] for ap in range(self.n_acts[i])
            )

            self.q_tables[i][(obs, a)] = q_sa + self.lr * (r + self.gamma * max_next - q_sa)

    def schedule_hyperparameters(self, t, max_t):
        """
        Linearly decays epsilon from 1.0 to 0.05 over 80% of the training duration.
        """
        decay_steps = 0.80 * max_t 
        if t < decay_steps:
            self.epsilon = 1.0 - (t / decay_steps) * 0.95
        else:
            self.epsilon = 0.05

# CQL Agent
class CQL:
    """
    Centralized Q-Learning Agent (Joint Action Learner).
    It maintains a single Q-table for the joint state and joint action.
    """
    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        gamma: float,
        learning_rate: float = 0.5,
        epsilon: float = 1.0,
        **kwargs,
    ):
        """
        Constructor of CQL

        Initializes variables for the centralized Q-learning agent, including
        generating the full joint action space.

        :param num_agents (int): number of agents
        :param action_spaces (List[Space]): action spaces of the environment for each agent
        :param gamma (float): discount factor (gamma)
        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for the agent

        :attr n_acts (List[int]): number of actions for each agent
        :attr q_table (DefaultDict): single table for Q-values mapping (joint_obs, joint_act) to Q-values
        :attr all_joint_actions (List[Tuple]): list of all possible joint actions (Cartesian product)
        """
        self.num_agents = num_agents
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # SINGLE Central Q-Table
        # Key: string representation of (joint_obs, joint_action_tuple)
        self.q_table = defaultdict(lambda: 0.0)

        # Compute all possible joint actions (Cartesian product)
        self.all_joint_actions = list(itertools.product(*[range(n) for n in self.n_acts]))

    def act(self, obss) -> List[int]:
        # Fix: Flatten numpy arrays to a hashable tuple, not a string
        # This handles the LBF observation format correctly
        joint_obs = tuple(o.astype(int).tobytes() for o in obss)

        if random.random() < self.epsilon:
            return list(random.choice(self.all_joint_actions))
        
        max_q = -float('inf')
        best_joint_actions = []

        for joint_act in self.all_joint_actions:
            # Fix: Use the tuple key directly
            q_val = self.q_table[(joint_obs, joint_act)]
            if q_val > max_q:
                max_q = q_val
                best_joint_actions = [joint_act]
            elif q_val == max_q:
                best_joint_actions.append(joint_act)
        
        selected_joint_action = random.choice(best_joint_actions)
        return list(selected_joint_action)

    def learn(self, obss, actions, rewards, n_obss, done):
        # Fix: Same tuple conversion for learning
        joint_obs = tuple(o.astype(int).tobytes() for o in obss)
        joint_n_obs = tuple(o.astype(int).tobytes() for o in n_obss)
        joint_action = tuple(actions)

        total_reward = sum(rewards)

        # Fix: Remove str() wrapping
        q_sa = self.q_table[(joint_obs, joint_action)]

        if done:
            max_next_q = 0.0
        else:
            max_next_q = -float('inf')
            for next_joint_act in self.all_joint_actions:
                # Fix: Remove str() wrapping
                q_val = self.q_table[(joint_n_obs, next_joint_act)]
                if q_val > max_next_q:
                    max_next_q = q_val
            
            if max_next_q == -float('inf'):
                max_next_q = 0.0

        updated_q = q_sa + self.learning_rate * (total_reward + self.gamma * max_next_q - q_sa)
        self.q_table[(joint_obs, joint_action)] = updated_q

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """
        Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        # Decay epsilon over the first 80% of steps, then stay at 0.05
        decay_steps = 0.8 * max_timestep
        if timestep < decay_steps:
             self.epsilon = 1.0 - (timestep / decay_steps) * 0.95
        else:
             self.epsilon = 0.05

# Training function
def train(agent_class, env, episodes=4000):
    """
    Main training loop.

    :param agent_class: Class reference (IQL or CQL) to instantiate
    :param env: Gymnasium environment instance
    :param episodes: Total number of episodes to train
    :return: (trained_agent, returns_array)
    """
    # Instantiate the agent (IQL or CQL) with default hyperparameters
    agent = agent_class(
        num_agents=env.unwrapped.n_agents,
        action_spaces=env.action_space,
        gamma=0.99,
        learning_rate=0.1,
        epsilon=1.0, # Start with 100% exploration
    )

    returns = []
    step_counter = 0
    # Calculate total max steps to define the epsilon decay timeline
    max_steps = episodes * env.unwrapped._max_episode_steps

    for ep in range(episodes):
        obss, _ = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            # Decay exploration rate (epsilon) based on total training progress
            agent.schedule_hyperparameters(step_counter, max_steps)

            # Select actions:
            # - IQL will treat 'obss' as N separate local observations
            # - CQL will squash 'obss' into a single Joint State inside .act()
            actions = agent.act(obss)

            # Step the environment
            n_obss, rewards, done, _, _ = env.step(actions)

            # Update Q-Tables:
            # - IQL updates N tables independently
            # - CQL updates based on the Joint State
            agent.learn(obss, actions, rewards, n_obss, done)

            # Accumulate team reward
            ep_return += sum(rewards)

            obss = n_obss
            step_counter += 1

        returns.append(ep_return)

    return agent, np.array(returns)

# Utilities
def save_returns(filename, returns):
    """
    Saves the episode returns to a CSV file.
    """
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/{filename}", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return"])
        for i, r in enumerate(returns):
            writer.writerow([i, r])

def plot_mean_returns(iql_returns, cql_returns, window=100):
    """
    Plots the rolling mean return for IQL and CQL comparison.
    """
    iql_mean = np.convolve(iql_returns, np.ones(window)/window, mode="valid")
    cql_mean = np.convolve(cql_returns, np.ones(window)/window, mode="valid")

    plt.figure(figsize=(8, 4))
    plt.plot(iql_mean, label="IQL (mean)")
    plt.plot(cql_mean, label="CQL (mean)")
    plt.xlabel("Episodes")
    plt.ylabel("Mean Episode Return")
    plt.title("Mean Episode Returns on LBF (5x5, 2p, 1f)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Video Recording Function
def record_video(agent, env_id, filename="video.gif", fps=4): 
    """
    Records a video (GIF) of the trained agent using direct Pyglet buffer capture.
    Videos are saved in this format since .mp4 led to exportation problems (first frames were skipped).
    
    :param agent: Trained agent instance
    :param env_id: Environment ID string
    :param filename: Output filename
    :param fps: Frames per second for the output GIF
    """
    print(f"Starting recording for {filename}...")
    
    # Initialize with render_mode=None
    env = gym.make(env_id, render_mode=None)
    
    # Force greedy policy
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    frames = []
    obs, info = env.reset()
    
    # Enable rendering and WARM UP the window
    env.unwrapped.render_mode = "human"
    env.render() # Trigger window opening
    time.sleep(0.5) # Wait for window to fully initialize
    
    # Capture "Frame 0" (Initial State)
    try:
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        image_data = buffer.get_image_data()
        arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
        arr = arr.reshape(buffer.height, buffer.width, 4)
        arr = arr[::-1, :, 0:3]
        frames.append(arr)
    except Exception as e:
        print(f"Warning: Could not capture Frame 0: {e}")

    try:
        done = False
        step = 0
        while not done and step < 50:
            # Agent acts
            actions = agent.act(obs)
            obs, rewards, done, truncated, info = env.step(actions)
            
            # Update Window
            env.render()

            # time.sleep(0.02)
            
            # Capture Frame
            try:
                buffer = pyglet.image.get_buffer_manager().get_color_buffer()
                image_data = buffer.get_image_data()
                arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
                arr = arr.reshape(buffer.height, buffer.width, 4)
                arr = arr[::-1, :, 0:3]
                frames.append(arr)
            except Exception:
                pass # Skip bad frames

            if done or truncated:
                print(f"Episode finished at step {step+1} with reward {sum(rewards)}")
                break
            
            step += 1
            # time.sleep(0.01) # Slow down execution slightly to ensure render keeps up

    finally:
        env.close()
        agent.epsilon = original_epsilon

        # Save video
        if len(frames) > 0:
            # Save as GIF
            print(f"Saving {filename}...")
            imageio.mimsave(
                filename, 
                frames, 
                fps=fps, 
                loop=0 # 0 means loop forever
            )
            print(f"Saved {filename}")


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    COOPERATION = False # Change based on the desired environment

    if COOPERATION:
        ENV_ID = "Foraging-5x5-2p-2f-coop-v3"
        num_episodes = 100000
    else:
        ENV_ID = "Foraging-5x5-2p-1f-v3"
        num_episodes = 50000

    env = gym.make(ENV_ID)

    print("Training IQL...")
    trained_iql, iql_returns = train(IQL, env, episodes=num_episodes)
    save_returns("iql_returns.csv", iql_returns)

    print("Training CQL...")
    trained_cql, cql_returns = train(CQL, env, episodes=num_episodes)
    save_returns("cql_returns.csv", cql_returns)

    plot_mean_returns(iql_returns, cql_returns)

    print("Recording trained agents...")
    record_video(trained_iql, ENV_ID, filename="iql_agent.gif")
    record_video(trained_cql, ENV_ID, filename="cql_agent.gif")

    env.close()