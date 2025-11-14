from collections import Counter
import numpy as np
from .pderl_tools import distilation_crossover, proximal_mutate, rl_to_evo

def crowding_distance_sort(indices_in_pareto, fitness, max_point=1e6):
    indices_in_pareto = np.array(indices_in_pareto)
    fitness = np.array(fitness)
    n_objectives = fitness.shape[1]
    pareto_value = fitness[indices_in_pareto]
    sort_obj = np.sort(pareto_value, axis=0).T
    argsort_obj = np.argsort(pareto_value, axis=0).T
    crowding_point = np.zeros(len(pareto_value))
    for obj in range(n_objectives):
        fmin, fmax = sort_obj[obj][0], sort_obj[obj][-1]
        if max_point > 0:
            crowding_point[argsort_obj[obj][0]] = max_point
            crowding_point[argsort_obj[obj][-1]] = max_point
        else:
            crowding_point[argsort_obj[obj][0]] = (sort_obj[obj][1] - sort_obj[obj][0]) / (fmax-fmin)
            crowding_point[argsort_obj[obj][-1]] = (sort_obj[obj][-1] - sort_obj[obj][-2]) / (fmax-fmin)
        for i, index in enumerate(argsort_obj[obj]):
            if i == 0 or i == len(argsort_obj[obj]) - 1:
                continue
            crowding_point[index] += (sort_obj[obj][i+1] - sort_obj[obj][i-1]) / (fmax - fmin)
    return indices_in_pareto[crowding_point.argsort(axis=0)[::-1]]

def pareto_front_sort(fitness):
    solution_count = len(fitness)
    dominate_index = [[] for _ in range(solution_count)]
    dominated_index = [[] for _ in range(solution_count)]
    pareto_rank = np.zeros(solution_count, dtype=np.int32)
    pareto_fronts = [[] for _ in range(solution_count)]
    for i in range(solution_count):
        for j in range(i+1, solution_count):
            if dominate_check(fitness[i], fitness[j]):
                dominate_index[i].append(j)
                dominated_index[j].append(i)
            elif dominate_check(fitness[j], fitness[i]):
                dominate_index[j].append(i)
                dominated_index[i].append(j)
        if len(dominated_index[i]) == 0:
            pareto_fronts[0].append(i)
            pareto_rank[i] = 0
    current_front = 0
    while len(pareto_fronts[current_front]) > 0:
        next_front = current_front + 1
        for fitness_idx in pareto_fronts[current_front]:
            for dominated_by_fi in dominate_index[fitness_idx]:
                dominated_index[dominated_by_fi].remove(fitness_idx)
                if len(dominated_index[dominated_by_fi]) == 0:
                    pareto_fronts[next_front].append(dominated_by_fi)
                    pareto_rank[dominated_by_fi] = next_front
        current_front = next_front
    # while len(pareto_fronts) > current_front:
    #     pareto_fronts.pop(current_front)
    return pareto_fronts[:current_front]

def nsga2_sort(fitness, max_point):
    pareto_fronts = pareto_front_sort(fitness)
    sorted_pareto_fronts = []
    for front in pareto_fronts:
        sorted_pareto_fronts.append(list(crowding_distance_sort(indices_in_pareto=front, fitness=fitness, max_point=max_point)))
    return sorted_pareto_fronts

def dominate_check(a, b):
    """Check if first solution is dominate second solution or not

    Args:
        a (numpy.darray): solution a
        b (numpy.darray): solution b

    Returns:
        True: if a >= b, else False
    """
    if np.sum((a >= b).astype(np.int32)) == len(a) and np.sum((a > b).astype(np.int32)) > 0:
        return True
    return False


class NSGA:
    def __init__(self, args, rl_agents, evaluate) -> None:
        self.args = args
        self.rl_agents = rl_agents
        self.pop_size = args.pop_size
        self.evaluate = evaluate
        self.temp_check_dup = None

    def selection_tournament_mo(self, fitness, sorted_pareto_front, pareto_rank, individual_type, tournament_size=4):
        found = False
        dad_idx, mom_idx = None, None
        while not found:
            tournament = np.random.choice(len(fitness), tournament_size)
            pareto_rank_priority = pareto_rank[tournament]
            crowding_distance_priority = [sorted_pareto_front[p_rank].index(idx) for idx, p_rank in zip(tournament, pareto_rank_priority)]
            tournament_sorted = [x for x, _, _ in sorted(zip(tournament, pareto_rank_priority, crowding_distance_priority), key=lambda tup:(tup[1], tup[2]))]
            for i in range(tournament_size):
                for j in range(i, tournament_size):
                    lower = min(tournament_sorted[i], tournament_sorted[j])
                    higher = max(tournament_sorted[i], tournament_sorted[j])
                    key = f"{lower}-{higher}"
                    if self.temp_check_dup[key] == 0:
                        dad_idx, mom_idx = lower, higher
                        found = True
                        self.temp_check_dup[key] = 1
                        break
                if found: break      

        if pareto_rank[dad_idx] != pareto_rank[mom_idx]:
            weight_idx = dad_idx if pareto_rank[dad_idx] < pareto_rank[mom_idx] else mom_idx
            if individual_type[dad_idx] == individual_type[mom_idx]:
                type_idx = individual_type[dad_idx]
            else:
                type_idx = individual_type[dad_idx] if pareto_rank[dad_idx] > pareto_rank[mom_idx] else individual_type[mom_idx]
        else:
            rank = pareto_rank[dad_idx]
            weight_idx = dad_idx if sorted_pareto_front[rank].index(dad_idx) < sorted_pareto_front[rank].index(mom_idx) else mom_idx
            if individual_type[dad_idx] == individual_type[mom_idx]:
                # individual_type_set = list(set(individual_type))
                # individual_type_set.pop(individual_type_set.index(individual_type[dad_idx]))
                # type_idx = np.random.choice(individual_type_set)

                ## Fix select only boundary critics ##
                boundary_indices = list(range(self.args.num_objectives))
                if individual_type[dad_idx] in boundary_indices:
                    boundary_indices.pop(boundary_indices.index(individual_type[dad_idx]))
                type_idx = np.random.choice(boundary_indices)
            else:
                type_idx = individual_type[dad_idx] if sorted_pareto_front[rank].index(dad_idx) > sorted_pareto_front[rank].index(mom_idx) else individual_type[mom_idx]
        
        if weight_idx == dad_idx:
            return mom_idx, dad_idx, type_idx
        return dad_idx, mom_idx, type_idx
    
    def tournament_selection_gradient(self, fitness, sorted_pareto_front, pareto_rank, tournament_size=4):
        tournament = np.random.choice(len(fitness), tournament_size)
        pareto_rank_priority = pareto_rank[tournament]
        crowding_distance_priority = [sorted_pareto_front[p_rank].index(idx) for idx, p_rank in zip(tournament, pareto_rank_priority)]
        tournament_sorted = [x for x, _, _ in sorted(zip(tournament, pareto_rank_priority, crowding_distance_priority), key=lambda tup:(tup[1], tup[2]))]
        return tournament_sorted[0]

    def mopderl_step(self, archive, pop, fitness, individual_type: list, sorted_pareto_front, num_frames, logger):
        """Perform crossover => mutation => selection in population with nsga 2

        Args:
            pop (List[GeneticAgent]): individuals in population
            fitness (numpy.darray or list): fitness of population
        """
        pareto_rank = np.zeros(len(fitness), dtype=int)
        for i, front in enumerate(sorted_pareto_front):
            for index in front:
                pareto_rank[index] = i
                
        logger.info("Begin step...")
        self.temp_check_dup = Counter([])
        for i in range(self.args.max_child):
            dad_idx, mom_idx, type_idx = self.selection_tournament_mo(fitness[:self.pop_size], sorted_pareto_front, pareto_rank, individual_type) 
            selected_agent = self.rl_agents[type_idx]   
            pop.append(distilation_crossover(self.args, pop[dad_idx], pop[mom_idx], selected_agent.critic))
            individual_type.append(type_idx)

        for rl_agent_id, _ in enumerate(self.rl_agents):
            dad_idx = self.tournament_selection_gradient(fitness[:self.pop_size], sorted_pareto_front, pareto_rank)
            selected_agent = self.rl_agents[rl_agent_id]
            pop.append(distilation_crossover(self.args, pop[dad_idx], selected_agent, selected_agent.critic))
            # pop.append(distilation_crossover(self.args, selected_agent, pop[dad_idx], selected_agent.critic, focus=True))
            individual_type.append(rl_agent_id)
        
        

        current_pop_len = len(pop)
        for i, actor in enumerate(pop[:current_pop_len]):
            if np.random.rand() <= self.args.mutation_prob:
                pop.append(proximal_mutate(self.args, actor, mag=self.args.mutation_mag, need_clone=True))
                individual_type.append(individual_type[i])
        
        for rl_agent_id, agent in enumerate(self.rl_agents):
            pop.append(proximal_mutate(self.args, agent, mag=self.args.mutation_mag, need_clone=True))
            individual_type.append(rl_agent_id)

        new_fitness = np.zeros((len(pop), fitness.shape[1]), dtype=np.float32)

        for i in range(len(pop)):
            if not pop[i].yet_eval:
                for _ in range(self.args.num_evals):
                    episode_reward = self.evaluate(pop[i], is_render=False, is_action_noise=False, store_transition=True)
                    new_fitness[i] += episode_reward
                new_fitness[i] /= self.args.num_evals
            else:
                new_fitness[i] = fitness[i]

        hypervolume, sparsity, pareto_front_size = archive.update(pop, new_fitness, num_frames)
        logger.info("=>>>>>> Hypervolume: {}".format(hypervolume))
        logger.info("=>>>>>> Sparsity: {}".format(sparsity))
        logger.info("=>>>>>> Num in front: {}".format(pareto_front_size))

        stats = {"Hypervolume": hypervolume, "Sparsity": sparsity, "Pareto size": pareto_front_size}

        survivor_indices = list(self.epoch(new_fitness))
        for i in reversed(range(len(pop))):
            if i not in survivor_indices:
                pop.pop(i)
                individual_type.pop(i)        
        return new_fitness[sorted(survivor_indices)], stats

    def epoch(self, fitness):
        """Sort the population indices by pareto front and by crowding distance

        Args:
            fitness (numpy.darray (num_individuals, n_objectives)): fitness of the population
            only_pareto_front (bool, optional): only sort and return rank if False, else return remained indices after selection. Defaults to False.

        Returns:
            _type_: _description_
        """
        # logger.info("==============Epoch_sort================")
        sorted_pareto_front = nsga2_sort(fitness, 1e6)
        selected = []
        front = 0
        
        while len(selected) < self.pop_size:
            if len(selected) + len(sorted_pareto_front[front]) > self.pop_size:
                selected = selected + list(sorted_pareto_front[front][:(self.pop_size-len(selected))])
            else:
                selected = selected + sorted_pareto_front[front]
            front += 1
        return selected