import pickle
import numpy as np
from nsga2_tools import *
import os
import torch
from utils import calculate_hv, calculate_sparsity
import pandas as pd


class Archive:
    def __init__(self, args, archive_folder) -> None:
        self.init_archive_folder(archive_folder)
        self.archive_dict = {}
        self.temp_saving = {}
        self.temp_removing = []
        self.fitness_np = None
        self.log_metrics = {"Frame": [], "Generation": [], "Hypervolume": [], "Sparsity": [], "Spread_out": [], "Num_policies": []}
        self.generation_cnt = 0
        self.warm_up_frames = args.warm_up_frames
    
    def init_archive_folder(self, archive_folder):
        self.archive_folder = archive_folder
        if not os.path.exists(self.archive_folder):
            os.mkdir(self.archive_folder)
        self.weight_folder = os.path.join(self.archive_folder, 'weights')
        if not os.path.exists(self.weight_folder):
            os.mkdir(self.weight_folder)
        self.pareto_history = os.path.join(self.archive_folder, "pareto_history")
        if not os.path.exists(self.pareto_history):
            os.mkdir(self.pareto_history)
        self.pareto_history_id = os.path.join(self.archive_folder, "pareto_history_id")
        if not os.path.exists(self.pareto_history_id):
            os.mkdir(self.pareto_history_id)
        self.log_metrics_file = os.path.join(self.archive_folder, "metrics.csv")
        self.archive_file = os.path.join(self.archive_folder, 'archive.pkl')


    def update(self, population, fitness, num_frames):
        all_fitness = list(self.archive_dict.values())
        all_id = list(self.archive_dict.keys())
        temp_mapping = {}
        indices = []
        for i, agent in enumerate(population):
            temp_mapping[agent.id] = agent
            if agent.id not in all_id:
                indices.append(i)
                all_id.append(agent.id)
        if len(all_fitness) != 0:
            all_fitness = np.concatenate([all_fitness, fitness[indices]], axis=0)
        else:
            all_fitness = fitness[indices]

        first_pareto_front = pareto_front_sort(all_fitness)[0]
        for index, identity in enumerate(all_id):
            if index in first_pareto_front:
                if identity not in self.archive_dict:
                    self.archive_dict[identity] = all_fitness[index]
                    actor_save_path = os.path.join(self.weight_folder, str(identity)+"state_dict.pkl")
                    self.temp_saving[actor_save_path] = temp_mapping[identity].actor.state_dict()
            else:
                if identity in self.archive_dict:
                    self.archive_dict.pop(identity)
                    actor_save_path = os.path.join(self.weight_folder, str(identity)+"state_dict.pkl")
                    self.temp_removing.append(actor_save_path)
        self.fitness_np = all_fitness[first_pareto_front]
        
        # hv = calculate_hv(self.fitness_np * (-1))
        hv = 808 # NOTE: HV calculation disabled temporarily to find the right reference point.
        sp = calculate_sparsity(self.fitness_np)
        spread_out = -1
        if self.fitness_np.shape[1] == 2:
            sorted_fitness = sorted(self.fitness_np, key=lambda x:x[0])
            spread_out = np.linalg.norm(np.array(sorted_fitness[0]) - np.array(sorted_fitness[-1])) 
        
        self.log_metrics["Frame"].append(np.max(num_frames - self.warm_up_frames) + self.warm_up_frames * self.fitness_np.shape[1])
        self.log_metrics["Generation"].append(self.generation_cnt)
        self.log_metrics["Hypervolume"].append(hv)
        self.log_metrics["Sparsity"].append(sp)
        self.log_metrics["Spread_out"].append(spread_out)
        pareto_front_size = len(self.fitness_np)
        self.log_metrics["Num_policies"].append(pareto_front_size)
        return hv, sp, pareto_front_size


    def save_info(self):
        # Saving dict for updating archive
        with open(self.archive_file, "wb") as f:
            pickle.dump(self.archive_dict, f)
        
        for path, state_dict in self.temp_saving.items():
            torch.save(state_dict, path)
        self.temp_saving.clear()

        for path in self.temp_removing:
            os.remove(path)
        self.temp_removing.clear()
        

        # Logging hv, sp, pareto
        pd.DataFrame(self.log_metrics).to_csv(self.log_metrics_file, index=False)

        # Logging pareto front score
        archive_ids = np.array(list(self.archive_dict.keys())).reshape(-1, 1)
        concat_info = np.concatenate((archive_ids, self.fitness_np), axis=1)
        sorted_by_first_obj = np.array(list(sorted(concat_info, key = lambda x:x[1])))
        pd.DataFrame(self.fitness_np).to_csv(os.path.join(self.pareto_history, "generation_{}.csv".format(self.generation_cnt)), index=False)
        pd.DataFrame(sorted_by_first_obj).to_csv(os.path.join(self.pareto_history_id, "generation_{}.csv".format(self.generation_cnt)), index=False)
        self.generation_cnt += 1

        

    def load_info(self):
        if os.path.exists(self.archive_file):
            with open(self.archive_file, "rb") as f:
                self.archive_dict = pickle.load(f)
        
        """Checking"""
        # dict_keys = np.array(sorted(list(self.archive_dict.keys())))
        # save_file = np.array(sorted([int(s[:-14]) for s in os.listdir(self.weight_folder)]))
        # if np.sum(np.abs(dict_keys - save_file)) != 0:
        #     print(dict_keys)
        #     print(save_file)
        #     raise AssertionError("Save fail") 

        if os.path.exists(self.log_metrics_file):
            old_metrics = pd.read_csv(self.log_metrics_file)
            self.log_metrics = {
                "Frame": list(old_metrics["Frame"]), 
                "Generation": list(old_metrics["Generation"]), 
                "Hypervolume": list(old_metrics["Hypervolume"]),
                "Sparsity": list(old_metrics["Sparsity"]),
                "Spread_out": list(old_metrics["Spread_out"]),
                "Num_policies": list(old_metrics["Num_policies"]),
            }
        self.generation_cnt = len(os.listdir(self.pareto_history))
        
        
