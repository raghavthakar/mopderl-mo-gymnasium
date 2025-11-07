"""
Evaluate Warm-Up Checkpoints.

This script loads the genetic agent populations saved at the end of the warm-up
phase from a specified run and evaluates their performance on the environment.

It provides a snapshot of the specialized policies before the multi-objective
stage begins.
"""
import os
import sys
import argparse
import numpy as np
import torch
import mo_gymnasium as gym
import mo_gymnasium.envs.mujoco  # Registers the environments
from pathlib import Path

# --- Import from your existing codebase ---
from parameters import Parameters
from ddpg import GeneticAgent
from utils import NormalizedActions


def evaluate_agent(agent, env, parameters, num_evals=10):
    """
    A standalone function to evaluate a single agent's performance.
    This logic is adapted from MOAgent.evaluate to avoid instantiating a full agent.
    """
    # CORRECTED: Get the number of objectives from the parameters object.
    total_reward = np.zeros(parameters.num_objectives, dtype=np.float32)
    seed = 2024
    for x in range(num_evals):
        state, _ = env.reset(seed=seed+x)
        done = False
        while not done:
            action = agent.actor.select_action(np.array(state))
            next_state, reward, terminated, truncated, _ = env.step(action.flatten())
            done = terminated or truncated
            total_reward += reward
            state = next_state
    return total_reward / num_evals


# --- Main Script ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agent populations from a warm-up checkpoint.")
    # --- Primary Arguments for this script ---
    parser.add_argument('-env', help='Environment name (e.g., MO-Swimmer-v2)', required=True, type=str)
    parser.add_argument('-logdir', help='Base folder where experiment results are saved', type=str, required=True)
    parser.add_argument('-run_id', help="The specific run ID to evaluate (e.g., 0 for 'run_0')", type=int, required=True)
    parser.add_argument('-seed', help='Random seed', type=int, default=2024)
    
    # --- Add ALL compatibility arguments for the Parameters class ---
    # These have default values and are hidden from the help message.
    parser.add_argument('-disable_cuda', action='store_true', default=True, help=argparse.SUPPRESS)
    parser.add_argument('-warm_up', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('-max_frames', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('-num_individuals', type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument('-rl_type', type=str, default='ddpg', help=argparse.SUPPRESS)
    parser.add_argument('-priority_mag', type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument('-boundary_only', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('-mut_mag', type=float, default=0.05, help=argparse.SUPPRESS)
    parser.add_argument('-mut_noise', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('-disable_wandb', action='store_true', default=True, help=argparse.SUPPRESS)
    parser.add_argument('-checkpoint', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('-checkpoint_id', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('-save_ckpt', type=int, default=0, help=argparse.SUPPRESS)

    # The Parameters class will parse all arguments, including the hidden ones.
    parameters = Parameters(parser)

    # Map old env names to new mo-gymnasium names
    name_map = {
        'MO-Swimmer-v2': 'mo-swimmer-v5',
        'MO-HalfCheetah-v2': 'mo-halfcheetah-v5',
        'MO-Hopper-v2': 'mo-hopper-v5',
        'MO-Walker2d-v2': 'mo-walker2d-v5',
        'MO-Ant-v2': 'mo-ant-v5',
    }

    # --- 1. Setup Environment ---
    print(f"Setting up environment for {parameters.env_name}...")
    try:
        env_name = name_map[parameters.env_name]
    except KeyError:
        print(f"Error: Environment '{parameters.env_name}' not found in name_map.")
        sys.exit(1)

    env = NormalizedActions(gym.make(env_name))

    # --- Set state and action dimensions on the parameters object ---
    parameters.state_dim = env.observation_space.shape[0]
    parameters.action_dim = env.action_space.shape[0]

    # --- 2. Locate Checkpoint Folder ---
    run_folder = Path(parameters.save_foldername) / parameters.env_name / f"run_{parameters.run_id}"
    warmup_checkpoint_folder = run_folder / "checkpoint" / "warm_up"

    if not warmup_checkpoint_folder.exists():
        print(f"Error: Could not find warm-up checkpoint folder at: {warmup_checkpoint_folder}")
        sys.exit(1)

    print(f"Found checkpoint folder: {warmup_checkpoint_folder}")

    # Determine the number of sub-populations by checking folder names
    pop_folders = sorted([d for d in warmup_checkpoint_folder.iterdir() if d.is_dir() and d.name.startswith('pop')])
    num_sub_populations = len(pop_folders)

    if num_sub_populations == 0:
        print("Error: No population folders ('pop0', 'pop1', etc.) found in the checkpoint directory.")
        sys.exit(1)

    print(f"Found {num_sub_populations} sub-populations to evaluate.")

    # --- 3. Load and Evaluate Populations ---
    all_results = []
    for i in range(num_sub_populations):
        pop_folder_path = warmup_checkpoint_folder / f"pop{i}"

        if not pop_folder_path.exists():
            print(f"Warning: Population folder {pop_folder_path} not found. Skipping.")
            continue

        agent_folders = sorted([d for d in pop_folder_path.iterdir() if d.is_dir()], key=lambda x: int(x.name))

        print(f"\n--- Evaluating Population {i} ({len(agent_folders)} agents) ---")

        population_results = []
        for agent_folder in agent_folders:
            agent_id = agent_folder.name
            agent = GeneticAgent(parameters)
            agent.load_info(str(agent_folder))

            # Evaluate the loaded agent over 3 runs for stability
            # CORRECTED: Pass the parameters object to the evaluation function.
            fitness = evaluate_agent(agent, env, parameters, num_evals=10)
            population_results.append(fitness)
            print(f"  Agent {agent_id}: Fitness = [Forward: {fitness[0]:.2f}, Energy Cost: {fitness[1]:.2f}]")

        all_results.append(np.array(population_results))

    # --- 4. Print Summary Statistics ---
    print("\n--- Summary Statistics ---")
    for i, results in enumerate(all_results):
        mean_fitness = np.mean(results, axis=0)
        std_fitness = np.std(results, axis=0)
        print(f"Population {i}:")
        print(f"  Mean Fitness: [Forward: {mean_fitness[0]:.2f}, Energy Cost: {mean_fitness[1]:.2f}]")
        print(f"  Std Dev:      [Forward: {std_fitness[0]:.2f}, Energy Cost: {std_fitness[1]:.2f}]")

    env.close()