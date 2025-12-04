from .parameters import Parameters
from . import ddpg 
from typing import List
import torch.distributions as dist
import fastrand, random
import numpy as np
import torch

def distilation_crossover(args, gene1, gene2, selected_critic, scalar_weight, focus=False):
    """
    Performs crossover by distilling the policy of parents into a new agent.
    
    Args:
        selected_critic: The MOCritic facade instance from the island agent.
        scalar_weight: The scalarization vector used to condition/route the critic.
    """
    new_agent = ddpg.GeneticAgent(args)
    if not focus:
        new_agent.buffer.add_latest_from(gene1.buffer, args.individual_bs // 2)
        new_agent.buffer.add_latest_from(gene2.buffer, args.individual_bs // 2)
    else:
        new_agent.buffer.add_latest_from(gene1.buffer, args.individual_bs)
    new_agent.buffer.shuffle()

    ddpg.hard_update(new_agent.actor, gene2.actor)
    batch_size = min(128, len(new_agent.buffer))
    iters = len(new_agent.buffer) // batch_size

    for epoch in range(12):
        for i in range(iters):
            batch = new_agent.buffer.sample(batch_size)
            # We pass the MOCritic facade and the weight. The GeneticAgent.update_parameters 
            # method will use the facade to route the input correctly.
            new_agent.update_parameters(batch, gene1.actor, gene2.actor, selected_critic, scalar_weight)
            
    return new_agent

def rl_to_evo(rl_agent, evo_net):
    for target_param, param in zip(evo_net.actor.parameters(), rl_agent.actor.parameters()):
        target_param.data.copy_(param.data)
    evo_net.buffer.reset()
    evo_net.buffer.add_content_of(rl_agent.buffer)

def proximal_mutate(args, gene, mag, need_clone=False):
    # Based on code from https://github.com/uber-research/safemutations 
    if need_clone:
        new_clone_gene = ddpg.GeneticAgent(args)
        clone(gene, new_clone_gene)
    else:
        new_clone_gene = gene
    model = new_clone_gene.actor

    batch = new_clone_gene.buffer.sample(min(args.mutation_batch_size, len(new_clone_gene.buffer)))
    state, _, _, _, _ = batch
    output = model(state)

    params = model.extract_parameters()
    tot_size = model.count_parameters()
    num_outputs = output.size()[1]

    if args.mutation_noise:
        mag_dist = dist.Normal(args.mutation_mag, 0.02)
        mag = mag_dist.sample()

    # initial perturbation
    normal = dist.Normal(torch.zeros_like(params), torch.ones_like(params) * mag)
    delta = normal.sample()

    # we want to calculate a jacobian of derivatives of each output's sensitivity to each parameter
    jacobian = torch.zeros(num_outputs, tot_size).to(args.device)
    grad_output = torch.zeros(output.size()).to(args.device)

    # do a backward pass for each output
    for i in range(num_outputs):
        model.zero_grad()
        grad_output.zero_()
        grad_output[:, i] = 1.0

        output.backward(grad_output, retain_graph=True)
        jacobian[i] = model.extract_grad()

    # summed gradients sensitivity
    scaling = torch.sqrt((jacobian**2).sum(0))
    scaling[scaling == 0] = 1.0
    scaling[scaling < 0.01] = 0.01
    delta /= scaling
    new_params = params + delta

    model.inject_parameters(new_params)
    new_clone_gene.yet_eval = False
    if need_clone:
        return new_clone_gene

def clone(master, replacee):  # Replace the replacee individual with master
    for target_param, source_param in zip(replacee.actor.parameters(), master.actor.parameters()):
        target_param.data.copy_(source_param.data)
    replacee.buffer.reset()
    replacee.buffer.add_content_of(master.buffer)

class PDERLTool:
    def __init__(self, args: Parameters, rl_agents, evaluate) -> None:
        self.args = args
        self.rl_agents = rl_agents
        self.evaluate = evaluate
        self.each_pop_size = args.each_pop_size
        self.num_elitists = int(args.elite_fraction * self.each_pop_size)
    
    @staticmethod
    def sort_groups_by_fitness(genomes, fitness):
        groups = []
        for i, first in enumerate(genomes):
            for second in genomes[i+1:]:
                if fitness[first] < fitness[second]:
                    groups.append((second, first, fitness[first] + fitness[second]))
                else:
                    groups.append((first, second, fitness[first] + fitness[second]))
        return sorted(groups, key=lambda group: group[2], reverse=True)


    @staticmethod
    def selection_tournament_pderl(index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings_indices = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings_indices.append(index_rank[winner])

        offsprings_indices = list(set(offsprings_indices))  # Find unique offsprings
        if len(offsprings_indices) % 2 != 0:  # Number of offsprings should be even
            offsprings_indices.append(offsprings_indices[fastrand.pcg32bounded(len(offsprings_indices))])
        return offsprings_indices
    
    def pderl_step(self, pop, rl_agent_id, fitness, logger):
        selected_agent = self.rl_agents[rl_agent_id]
        logger.info("=>>>>>> Rl_agent_id: " + str(rl_agent_id))
        logger.info("=>>>>>> Fitness: ")
        logger.info(str(fitness))
        scalared_fitness = np.dot(fitness, selected_agent.scalar_weight)
        index_rank = np.argsort(scalared_fitness)[::-1]
        elitist_indices = index_rank[:self.num_elitists]
        active_critic = selected_agent.critic

        # Selection step
        offsprings_indices = self.selection_tournament_pderl(index_rank, num_offsprings=len(index_rank) - self.num_elitists, tournament_size=3)
        
        # Unselected candidates
        unselects_indices = []
        new_elitist_indices = []
        for index in range(self.each_pop_size):
            if index not in elitist_indices  and index not in offsprings_indices:
                unselects_indices.append(index)
        random.shuffle(unselects_indices)

        for i in elitist_indices:
            try: replacee = unselects_indices.pop(0)
            except: replacee = offsprings_indices.pop(0)
            new_elitist_indices.append(replacee)
            clone(master=pop[i], replacee=pop[replacee])

        fitness_sorted_group = PDERLTool.sort_groups_by_fitness(new_elitist_indices+offsprings_indices, scalared_fitness)
        for i, unselected_index in enumerate(unselects_indices):
            first, second, _ = fitness_sorted_group[i % len(fitness_sorted_group)]
            if scalared_fitness[first] < scalared_fitness[second]:
                first, second = second, first
            clone(distilation_crossover(self.args, pop[first], pop[second], active_critic, selected_agent.scalar_weight), pop[unselected_index])
        
        # Crossover for selected offsprings
        for i in offsprings_indices:
            if random.random() < self.args.crossover_prob:
                others = offsprings_indices.copy()
                others.remove(i)
                off_j = random.choice(others)
                clone(distilation_crossover(self.args, pop[i], pop[off_j], active_critic, selected_agent.scalar_weight), pop[i])

        for i in range(self.each_pop_size):
            if i not in new_elitist_indices:  # Spare the new elitists
                if random.random() < self.args.mutation_prob:
                    proximal_mutate(self.args, pop[i], mag=self.args.mutation_mag)
        return