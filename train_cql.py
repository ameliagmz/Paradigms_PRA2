import copy
import random
import itertools
from collections import defaultdict
from typing import List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Space
from gymnasium.spaces.utils import flatdim

from utils import (
    visualise_q_tables_cql,
    visualise_q_convergence_cql,
    visualise_evaluation_returns,
)
from matrix_game import create_pd_game


class CQL:
    """
    Centralized Q-Learning Agent (Joint Action Learner).
    It maintains a SINGLE Q-table for the joint state and joint action.
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
        """
        Implement the epsilon-greedy action selection for the JOINT action space

        :param obss (List): list of observations for each agent
        :return (List[int]): index of selected action for each agent (constituting the joint action)
        """
        joint_obs = tuple(obss)

        # Epsilon-greedy on the JOINT action space
        if random.random() < self.epsilon:
            # Random joint action
            return list(random.choice(self.all_joint_actions))
        
        # Greedy selection: Find the joint action with the highest Q-value
        max_q = -float('inf')
        best_joint_actions = []

        # Iterate through every possible combination of actions to find the best team move
        for joint_act in self.all_joint_actions:
            q_val = self.q_table[str((joint_obs, joint_act))]
            if q_val > max_q:
                max_q = q_val
                best_joint_actions = [joint_act]
            elif q_val == max_q:
                best_joint_actions.append(joint_act)
        
        # Pick one of the best joint actions randomly
        selected_joint_action = random.choice(best_joint_actions)
        return list(selected_joint_action)

    def learn(
        self,
        obss: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        n_obss: List[np.ndarray],
        done: bool,
    ):
        """
        Updates the single Q-table based on the joint experience and total team reward

        :param obss (List[np.ndarray]): list of observations for each agent
        :param actions (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param n_obss (List[np.ndarray]): list of observations after taking the action for each agent
        :param done (bool): flag indicating whether a terminal state has been reached
        :return (None): updated Q-value for the current joint action
        """
        joint_obs = tuple(obss)
        joint_n_obs = tuple(n_obss)
        joint_action = tuple(actions)

        # Centralized agents usually optimize the sum of rewards (Team Reward)
        total_reward = sum(rewards)

        q_sa = self.q_table[str((joint_obs, joint_action))]

        # Calculate Max Q for next state (over all possible joint actions)
        if done:
            # If the episode has ended, there is no future value
            max_next_q = 0.0
        else:
            # Calculate max_a' Q(s', a')
            # We look up the Q-value for the NEXT joint observation (joint_n_obs)
            # paired with every possible NEXT joint action, and take the maximum.
            max_next_q = -float('inf')
            for next_joint_act in self.all_joint_actions:
                q_val = self.q_table[str((joint_n_obs, next_joint_act))]
                if q_val > max_next_q:
                    max_next_q = q_val
            
            # Safety if q-table is empty/new
            if max_next_q == -float('inf'):
                max_next_q = 0.0

        # Apply Bellman update equation
        updated_q = q_sa + self.learning_rate * (total_reward + self.gamma * max_next_q - q_sa)
        # Update Q-table with the new value
        self.q_table[str((joint_obs, joint_action))] = updated_q

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


CONFIG = {
    "seed": 42,
    "gamma": 0.99,
    "total_eps": 5000, # Reduced eps needed for Matrix games
    "ep_length": 1,
    "eval_freq": 100,
    "lr": 0.1,
    "init_epsilon": 1.0,
    "eval_epsilon": 0.0, # Greedy evaluation
}

def cql_eval(env, config, q_table, eval_episodes=500, output=True):
    """
    Evaluates the Centralized CQL agent using the trained Q-table

    :param env: the environment to evaluate on
    :param config (dict): configuration dictionary containing parameters like gamma and lr
    :param q_table (DefaultDict): the trained Q-table to evaluate
    :param eval_episodes (int): number of episodes to run for evaluation
    :param output (bool): whether to print the evaluation results
    :return (Tuple[np.ndarray, np.ndarray]): mean and standard deviation of returns
    """
    # Initialize agent with evaluation parameters
    eval_agents = CQL(
        num_agents=env.n_agents,
        action_spaces=env.action_space,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        epsilon=config["eval_epsilon"],
    )
    eval_agents.q_table = q_table # Load the trained Q-table

    episodic_returns = []
    for _ in range(eval_episodes):
        obss, _ = env.reset()
        episodic_return = np.zeros(env.n_agents)
        done = False

        while not done:
            actions = eval_agents.act(obss) # Select greedy actions using trained Q-table
            obss, rewards, done, _, _ = env.step(actions)
            episodic_return += rewards # Accumulate rewards

        episodic_returns.append(episodic_return)

    # Calculate statistics across all evaluation episodes
    mean_return = np.mean(episodic_returns, axis=0)
    std_return = np.std(episodic_returns, axis=0)

    if output:
        print("EVALUATION RETURNS:")
        print(f"\tAgent 1: {mean_return[0]:.2f} ± {std_return[0]:.2f}")
        print(f"\tAgent 2: {mean_return[1]:.2f} ± {std_return[1]:.2f}")
        print(f"\tTotal Team: {np.sum(mean_return):.2f}")
    return mean_return, std_return

def train(env, config, output=True):
    """
    Training loop for the Centralized CQL agent

    :param env: the environment to train on
    :param config (dict): configuration dictionary containing training parameters
    :param output (bool): whether to print training progress
    :return (Tuple): tuple containing lists of mean returns, std returns, 
        snapshots of Q-tables, and the final Q-table
    """
    # Initialize the Centralized Agent
    agents = CQL(
        num_agents=env.n_agents,
        action_spaces=env.action_space,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        epsilon=config["init_epsilon"],
    )

    step_counter = 0
    max_steps = config["total_eps"] * config["ep_length"]

    evaluation_return_means = []
    evaluation_return_stds = []
    
    # Store snapshots of the single centralized Q-table
    evaluation_q_tables = []

    print("Starting Training with Centralized CQL (Joint Action Learning)...")

    for eps_num in range(config["total_eps"]):
        obss, _ = env.reset()
        done = False

        while not done:
            # Decay epsilon based on progress
            agents.schedule_hyperparameters(step_counter, max_steps)

            # Select joint action
            acts = agents.act(obss)
            n_obss, rewards, done, _, _ = env.step(acts)
            
            # Update table using joint actions and summed rewards
            agents.learn(obss, acts, rewards, n_obss, done)

            step_counter += 1
            obss = n_obss

        # Periodic Evaluation
        if eps_num > 0 and eps_num % config["eval_freq"] == 0:
            mean_return, std_return = cql_eval(
                env, config, agents.q_table, output=output
            )
            evaluation_return_means.append(mean_return)
            evaluation_return_stds.append(std_return)
            # Deepcopy to save the state of the Q-table at this specific timestep
            evaluation_q_tables.append(copy.deepcopy(agents.q_table))

    return (
        evaluation_return_means,
        evaluation_return_stds,
        evaluation_q_tables,
        agents.q_table,
    )

if __name__ == "__main__":
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    
    env = create_pd_game()
    
    # Run training loop
    evaluation_return_means, evaluation_return_stds, eval_q_tables, final_q_table = train(env, CONFIG)

    # Visualize results
    visualise_q_tables_cql(final_q_table)
    visualise_evaluation_returns(evaluation_return_means, evaluation_return_stds)
    visualise_q_convergence_cql(eval_q_tables, env)