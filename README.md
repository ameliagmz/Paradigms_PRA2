# Multi-Agent Reinforcement Learning: IQL vs CQL

This repository implements **Independent Q-Learning (IQL)** and **Centralized Q-Learning (CQL)** across two distinct multi-agent environments: the Iterated Prisoner's Dilemma (Matrix Game) and Level-Based Foraging (Gridworld).

The project aims to demonstrate the differences in coordination and scalability between independent learners and centralized joint-action learners.

## 📦 Requirements

The dependencies for this project are split into two parts. You can install them using the provided text files:

```
pip install -r requirements_part1.txt
pip install -r requirements_part2.txt
```

Part 1: Core libraries for the Matrix Game (Task 1), including numpy, gymnasium, and matplotlib.

Part 2: Libraries for the Gridworld (Task 2), including lbforaging, imageio, and pyglet.

Note: If lbforaging is not found via pip, you may need to install it directly from the Level-Based Foraging repository.

## 📂 Project Structure
The codebase is organized by task, sharing common utilities where appropriate.

```
├── requirements_part1.txt # Dependencies for Task 1
├── requirements_part2.txt # Dependencies for Task 2
├── iql.py                 # Defines the IQL Agent class (used in Task 1)
├── train_iql.py           # Training script for IQL on Prisoner's Dilemma
├── train_cql.py           # Training script (and CQL class definition) for Prisoner's Dilemma
├── matrix_game.py         # Defines the MatrixGame environment logic
├── utils.py               # Visualization and evaluation utilities for Task 1
└── lbf.py                 # All-in-one script for Task 2 (Agents + Training + Visualization)
```

## 🕹️ Task 1: Iterated Prisoner's Dilemma
This task compares agents in a classic matrix game to test their ability to cooperate. The environment logic is defined in matrix_game.py.

### 1. Independent Q-Learning (IQL)
The IQL class is imported from iql.py. This script trains the agent to play the game and automatically generates evaluation plots.

```
python train_iql.py
```

### 2. Centralized Q-Learning (CQL)
For this task, the CQL class is defined directly within the training script. This agent uses a single joint Q-table to optimize the total team reward.

```
python train_cql.py
```

#### Outputs:
Both scripts utilize utils.py to generate and display plots showing training convergence and evaluation returns.

## 🍎 Task 2: Level-Based Foraging (LBF)
This task tests the agents in a gridworld environment where they must coordinate to collect food. The implementation uses a specific tabular approach adapted for the larger LBF state space, with modifications to handle partial observability.

#### Running the Experiment
The file lbf.py is self-contained. It includes:
- The modified IQL and CQL class definitions for gridworlds.
- The main training loop.
- Visualization and video recording logic.

To run the training and evaluation:

```
python lbf.py
```

#### Configuration
You can toggle the environment difficulty (e.g., whether agents must lift food together or can forage alone) by modifying the COOPERATION constant at the top of lbf.py.

```
# Inside lbf.py

# Set to True for environments where agents must load together (Foraging-5x5-2p-2f-coop-v3)
# Set to False for environments where agents can forage alone (Foraging-5x5-2p-1f-v3)
COOPERATION = True 
```

#### Outputs:
- Plots: A graph comparing the Mean Episode Returns of IQL vs. CQL over time.
- Logs: CSV files containing raw return data.
- Video: A .gif (or .mp4) recording of the trained agents acting in the environment, saved to the working directory.
