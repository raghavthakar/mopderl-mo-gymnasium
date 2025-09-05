import os, sys
from pathlib import Path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, ".."))

from datetime import datetime
import numpy as np, os, time, random
import mo_gymnasium as gym
import mo_gymnasium.envs.mujoco #  Importing this module registers the environments
import torch
import argparse
from parameters import Parameters
import logging
import mo_agent
import utils
import wandb

parser = argparse.ArgumentParser()
# Updated environment choices to reflect mo-gymnasium names
parser.add_argument('-env', help='Environment Choices: (MO-Swimmer-v2) (MO-HalfCheetah-v2) (MO-Hopper-v2) ' +
                                 '(MO-Walker2d-v2) (MO-Ant-v2)', required=True, type=str)
parser.add_argument('-seed', help='Random seed to be used', type=int, required=True)
parser.add_argument('-disable_cuda', help='Disables CUDA', action='store_true')
parser.add_argument('-mut_mag', help='The magnitude of the mutation', type=float, default=0.05)
parser.add_argument('-mut_noise', help='Use a random mutation magnitude', action='store_true')
parser.add_argument('-logdir', help='Folder where to save results', type=str, required=True)
parser.add_argument('-warm_up', help='Warm up frames', type=int)
parser.add_argument('-max_frames', help='Max frames', type=int)
parser.add_argument('-num_individuals', help='Number of individual per pderl population', type=int, default=10)
parser.add_argument('-num_generations', help='Max number of generation', type=int)
parser.add_argument('-priority_mag', help='Percent of priority for objective', type=float, default=1.0)
parser.add_argument('-rl_type', help='Type of rl-agents', type=str, default="ddpg")
parser.add_argument('-checkpoint', help='Load checkpoint', action='store_true')
parser.add_argument('-checkpoint_id', help='Select -run- to load checkpoint', type=int)
parser.add_argument('-run_id', help="Specify run id, if not given, get id as len(run)", type=int)
parser.add_argument('-save_ckpt', help="Save checkpoint every _ step, 0 for no save", type=int, default=1)
parser.add_argument('-disable_wandb', action="store_true", default=False)
parser.add_argument('-boundary_only', action='store_true', default=False)

# Updated map with correct mo-gymnasium environment names and versions
name_map = {
    'MO-Swimmer-v2': 'mo-swimmer-v5',
    'MO-HalfCheetah-v2': 'mo-halfcheetah-v5',
    'MO-Hopper-v2': 'mo-hopper-v5',
    'MO-Walker2d-v2': 'mo-walker2d-v5',
    'MO-Ant-v2': 'mo-ant-v5',
}

if __name__ == "__main__":
    parameters = Parameters(parser)  # Inject the cla arguments in the parameters object

    if not os.path.exists(parameters.save_foldername):
        os.mkdir(parameters.save_foldername)
    env_folder = os.path.join(parameters.save_foldername, parameters.env_name)
    if not os.path.exists(env_folder):
        os.mkdir(env_folder)
    list_run = sorted(os.listdir(env_folder))
    if parameters.checkpoint:
        if parameters.checkpoint_id is not None:
            run_folder = os.path.join(env_folder, "run_"+str(parameters.checkpoint_id))
        else:
            run_folder = os.path.join(env_folder, list_run[-1])
    else:
        run_id = "run_"+str(len(list_run))
        if parameters.run_id is not None:
            run_id = "run_"+str(parameters.run_id)
        run_folder = os.path.join(env_folder, run_id)
    if not os.path.exists(run_folder):    
        os.mkdir(run_folder)    

    if parameters.wandb: wandb.init(project=parameters.env_name, entity="mopderl", id=str(Path(run_folder).name), resume=parameters.checkpoint) 
    logging.basicConfig(filename=os.path.join(run_folder, "logger.log"),
                        format=('[%(asctime)s] - '
                                '[%(levelname)4s]:\t'
                                '%(message)s'
                                '\t(%(filename)s:'
                                '%(funcName)s():'
                                '%(lineno)d)\t'),
                        filemode='a',
                        level=logging.DEBUG)
    logger = logging.getLogger()
    logger.info("Start time: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    # Create Env
    env = utils.NormalizedActions(gym.make(name_map[parameters.env_name]))
    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]

    # Write the parameters to a the info file and print them
    parameters.write_params(path=run_folder)

    # Seed
    torch.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)
    random.seed(parameters.seed)

    # Create Agent
    reward_keys = utils.parse_json("reward_keys.json")[parameters.env_name]
    agent = mo_agent.MOAgent(parameters, env, reward_keys, run_folder)
    print('Running', parameters.env_name, ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim)
    logger.info('Running' + str(parameters.env_name) + ' State_dim:' + str(parameters.state_dim) + ' Action_dim:' + str(parameters.action_dim))
    logger.info("Priority: " + str(parameters.priority))

    time_start = time.time()
    warm_up_saved = False

    while np.sum(agent.num_frames < agent.max_frames).astype(int) > 0:
        logger.info("************************************************")
        logger.info("\t\tGeneration: " + str(agent.iterations))
        logger.info("************************************************")
        stats_wandb = agent.train_final(logger)

        if parameters.wandb and len(stats_wandb):
            current_pareto = stats_wandb.pop("pareto")
            current_pareto = [list(point) for point in current_pareto]
            table = wandb.Table(data=current_pareto, columns=reward_keys)
            wandb.log({ 
                **{"Current pareto front" : wandb.plot.scatter(table, reward_keys[0], reward_keys[1], title="Current pareto front")}, 
                **stats_wandb
            })

        print('#Generation:', agent.iterations, '#Frames:', agent.num_frames,
              ' ENV:  '+ parameters.env_name)
        print()
        logger.info("\n\n")
        logger.info("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        logger.info("=>>>>>> Num frames: " + str(agent.num_frames))
        logger.info("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        if (parameters.save_ckpt_period > 0 and agent.iterations % parameters.save_ckpt_period == 0) or \
            np.sum(agent.num_frames < agent.max_frames).astype(int) == 0:
            agent.save_info()
            logger.info("Save info successfully!\n\n")

        if not warm_up_saved and np.sum(agent.num_frames < parameters.warm_up_frames).astype(int) == 0:
            agent.save_info()
            logger.info("Save warmup infor successfully!!!\n\n")   
            warm_up_saved = True

        if len(stats_wandb):
            agent.archive.save_info()
        
    logger.info("End time: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))