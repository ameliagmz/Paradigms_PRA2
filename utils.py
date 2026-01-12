import matplotlib.pyplot as plt
import numpy as np

FIG_WIDTH=5
FIG_HEIGHT=2
FIG_ALPHA=0.2
FIG_WSPACE=0.3
FIG_HSPACE=0.2


def visualise_q_tables(q_tables):
    for i, q_table in enumerate(q_tables):
        print(f"Q-table for Agent {i + 1}:")
        for a in range(2):
            print(f"Q({a + 1}) = {q_table[str((0, a))]:.2f}")
        print()


def visualise_evaluation_returns(means, stds):
    """
    Plot evaluation returns

    :param means (List[List[float]]): mean evaluation returns for each agent
    :param stds (List[List[float]]): standard deviation of evaluation returns for each agent
    """
    n_agents = len(means[0])
    n_evals = len(means)

    fig, ax = plt.subplots(nrows=1, ncols=n_agents, figsize=(FIG_WIDTH + 1.0, FIG_HEIGHT * n_agents))

    colors = ["b", "r"]
    for i, color in enumerate(colors):
        ax[i].plot(range(n_evals), [mean[i] for mean in means], label=f"Agent {i+1}", color=color)
        ax[i].fill_between(range(n_evals), [mean[i] - std[i] for mean, std in zip(means, stds)],
                           [mean[i] + std[i] for mean, std in zip(means, stds)], alpha=FIG_ALPHA, color=color)
        ax[i].set_xlabel("Evaluations")
        ax[i].set_ylabel("Evaluation return")
    fig.legend()
    fig.subplots_adjust(hspace=FIG_HSPACE)

    plt.show()

def visualise_q_convergence(eval_q_tables, env, savefig=None):
    """
    Plot q_table convergence
    :param eval_q_tables (List[List[Dict[Act, float]]]): q_tables of both agents for each evaluation
    :param env (gym.Env): gym matrix environment with `payoff` attribute
    :param savefig (str): path to save figure
    """
    assert hasattr(env, "payoff")
    payoff = np.array(env.payoff)
    n_actions = 2
    n_agents = 2
    assert payoff.shape == (n_actions, n_actions, n_agents), "Payoff matrix must be 2x2x2 for 2x2 PD game"
    # (n_evals, n_agents, n_actions)
    q_tables = np.array(
            [[[q_table[str((0, act))] for act in range(n_actions)] for q_table in q_tables] for q_tables in eval_q_tables]
    )

    fig, ax = plt.subplots(nrows=n_agents, ncols=n_actions, figsize=(n_actions * FIG_WIDTH, FIG_HEIGHT * n_agents))

    for i in range(n_agents):
        max_payoff = payoff[:, :, i].max()
        min_payoff = payoff[:, :, i].min()
    
        for act in range(n_actions):
            # plot max Q-values
            if i == 0:
                max_r = payoff[act, :, i].max()
                max_label = rf"$max_b Q(a, b)$"
                q_label = rf"$Q(a_{act}, \cdot)$"
            else:
                max_r = payoff[:, act, i].max()
                max_label = rf"$max_a Q(a, b_{act})$"
                q_label = rf"$Q(\cdot, b_{act})$"
            ax[i, act].axhline(max_r, ls='--', color='r', alpha=0.5, label=max_label)

            # plot respective Q-values
            q_values = q_tables[:, i, act]
            ax[i, act].plot(q_values, label=q_label)

            # axes labels and limits
            ax[i, act].set_ylim([min_payoff - 0.05, max_payoff + 0.05])
            ax[i, act].set_xlabel(f"Evaluations")
            if i == 0:
                ax[i, act].set_ylabel(fr"$Q(a_{act})$")
            else:
                ax[i, act].set_ylabel(fr"$Q(b_{act})$")

            ax[i, act].legend(loc="upper center")

    fig.subplots_adjust(wspace=FIG_WSPACE)

    if savefig is not None:
        plt.savefig(f"{savefig}.pdf", format="pdf")

    plt.show()

def visualise_q_tables_cql(q_table):
    """
    Visualizes the single Q-table for a CQL agent.
    """
    print("\n--- Centralized Joint Q-Table ---")
    sorted_keys = sorted(q_table.keys())
    
    for key_str in sorted_keys:
        try:
            # Parse string key back to tuple: ((obs), (act1, act2))
            key_tuple = eval(key_str) 
            joint_obs, joint_act = key_tuple
            value = q_table[key_str]
            print(f"State: {joint_obs} | Joint Action: {joint_act} => Q-Value: {value:.2f}")
        except Exception:
            print(f"Key: {key_str} => {q_table[key_str]:.2f}")
    print()

def visualise_q_convergence_cql(eval_q_tables, env, joint_obs_tuple=(0, 0), savefig=None):
    """
    Plots the convergence of the JOINT Q-values (Team Rewards) for the CQL Agent.
    """
    assert hasattr(env, "payoff"), "Environment must have `payoff` attribute"
    payoff = np.array(env.payoff) # Shape: (n_act, n_act, n_agents)
    
    # The 4 possible joint actions in 2x2 game: (0,0), (0,1), (1,0), (1,1)
    joint_actions = [(0,0), (0,1), (1,0), (1,1)]
    history = {ja: [] for ja in joint_actions}

    # Extract history
    for q_table in eval_q_tables:
        for ja in joint_actions:
            key = str((joint_obs_tuple, ja))
            val = q_table.get(key, 0.0)
            history[ja].append(val)

    # Plot 2x2 grid
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(2 * FIG_WIDTH, 2 * FIG_HEIGHT))
    axes = axes.flatten() 

    for idx, ja in enumerate(joint_actions):
        ax = axes[idx]
        
        # Plot learned Q-value (Blue)
        ax.plot(history[ja], label=f"Q{ja}", color='blue')
        
        # Plot target team reward (Red dashed)
        act1, act2 = ja
        # Sum the rewards for both agents to get Team Reward
        team_reward = payoff[act1][act2][0] + payoff[act1][act2][1]
        
        ax.axhline(team_reward, ls="--", color="r", alpha=0.5, label=f"Target ({team_reward})")
        
        # ax.set_title(f"Joint Action {ja}")
        ax.set_xlabel("Evaluation Step")
        ax.set_ylabel("Joint Q-Value")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Convergence of Joint Q-Values (Starting State)")
    fig.subplots_adjust(wspace=FIG_WSPACE, hspace=FIG_HSPACE)

    if savefig:
        plt.savefig(f"{savefig}.pdf", format="pdf")
    plt.show()