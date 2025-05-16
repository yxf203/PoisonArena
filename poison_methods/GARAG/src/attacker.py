
import copy
import random

from textattack.shared import AttackedText

from textattack.constraints.pre_transformation import StopwordModification, MinWordLength
from textattack.transformations import WordSwapEmbedding
from textattack.shared import AttackedText
from textattack.constraints import PreTransformationConstraint

import os
from src.attack_module.constraint import LabelConstraint
from src.attack_module.goal import Double_GoalFunction
from src.attack_module.transformation import Typo
import json
from textattack.transformations import CompositeTransformation

from tqdm import tqdm

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import logging

import copy
import math
import random
import numpy as np
import time
from src.attack_module.constraint import LabelConstraint
from src.util import EM, f1, acc

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.misc import random_permuations
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.util.dominator import Dominator

import logging




logger = logging.getLogger(__name__)

attack_mapping = {
    "typo": Typo,
}

def build_attack(opt, dataset):

        if opt.method in attack_mapping:
            trans_cls = attack_mapping[opt.method]
        else:
            raise NotImplementedError()
    
        if opt.with_wordswap:
            wordswap = WordSwapEmbedding()

            transformation = CompositeTransformation(
                [
                    wordswap,
                    trans_cls()
                ]
            )
        else:
            transformation = trans_cls()

        constraints = [LabelConstraint(), MinWordLength(3), StopwordModification()]

        if opt.is_genetic:

            goal_function = Double_GoalFunction(opt)
            
            attacker = CustomGenetic(
                transformation=transformation,
                constraints=constraints,
                goal_function=goal_function,
                pct_words_to_swap=opt.perturbation_level,
                pop_size=opt.transformations_per_example,
                max_iters=opt.max_iters,
                not_cross=opt.not_cross,
                not_mut=opt.not_mut,
                not_sort=opt.not_sort
            )
            return attacker, dataset
        else:
            NotImplementedError()
    
def binary_tournament(pop, P):
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):

        a, b = P[i, 0], P[i, 1]
        
        f_a, cd_a = pop[a].get_scores(), pop[a].get_crowding()
        f_b, cd_b = pop[b].get_scores(), pop[b].get_crowding()

        rel = Dominator.get_relation(f_a, f_b)
        if rel == 1:
            S[i] = a
        elif rel == -1:
            S[i] = b
        # if rank or domination relation didn't make a decision compare by crowding
        if np.isnan(S[i]):
            S[i] = compare(a, cd_a, b, cd_b, method='larger_is_better', return_random_if_equal=True)

    return S[:, None].astype(int, copy=False)

class CustomTournament(TournamentSelection):
    def do(self, pop, n_select, n_parents):
        n_random = n_select * n_parents * self.pressure

        n_perms = math.ceil(n_random / len(pop))
        P = random_permuations(n_perms, len(pop))[:n_random]
        P = np.reshape(P, (n_select * n_parents, self.pressure))

        S = self.func_comp(pop, P)

        return np.reshape(S, (n_select, n_parents))

class CustomSurvival(RankAndCrowdingSurvival):

    def do(self,
            F,
            pop,
            fronts,
            n_survive=None):

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):
            
            I = np.arange(len(front))

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(I) > n_survive:

                # Define how many will be removed
                n_remove = len(survivors) + len(front) - n_survive

                # re-calculate the crowding distance of the front
                crowding_of_front = \
                    self.crowding_func.do(
                        F[front, :],
                        n_remove=n_remove
                    )

                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:-n_remove]

            # otherwise take the whole front unsorted
            else:
                # calculate the crowding distance of the front
                crowding_of_front = \
                    self.crowding_func.do(
                        F[front, :],
                        n_remove=0
                    )

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set_rank(k)
                pop[i].set_crowding(crowding_of_front[j])

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])
        return [pop[s] for s in survivors]

class Population:
    def __init__(self, initial_text):
        self.attacked_text= AttackedText(initial_text)
        self.modified_indices = None
        self.replacement_words = None
        self.og_scores = []
        self.scores = []
        self.rank = None
        self.crowding_distance = None

    def get_perturbed_text(self):
        assert self.modified_indices is not None
        assert self.replacement_words is not None
        return self.attacked_text.replace_words_at_indices(self.modified_indices, self.replacement_words).text

    def set_modified(self, words, indices):
        self.replacement_words = words
        self.modified_indices = indices
        return self

    def get_modified(self):
        assert self.replacement_words is not None and self.modified_indices is not None
        return self.replacement_words, self.modified_indices

    def get_modified_words(self):
        assert self.modified_indices is not None
        return self.replacement_words

    def get_modified_indices(self):
        assert self.modified_indices is not None
        return self.modified_indices
    
    def set_scores(self, scores):
        #self.scores.append(self.og_scores[0] / scores[0])
        self.scores.append(math.exp(self.og_scores[0] - scores[0]))
        self.scores.append(scores[1] / self.og_scores[1])
        return self

    def get_scores(self):
        return self.scores
    
    def get_metrics(self):
        return len(self.modified_indices)

    def set_rank(self, rank):
        self.rank = rank
        return self
    
    def get_rank(self):
        return self.rank

    def set_crowding(self, crowding):
        self.crowding_distance = crowding
        return self
    
    def get_crowding(self):
        return self.crowding_distance

class CustomGenetic:

    def __init__(
        self,
        transformation,
        constraints,
        goal_function,
        pop_size=50,
        max_iters=50,
        pct_words_to_swap=0.1,
        crossover_prob=0.2,
        mutation_prob=0.4,
        parents_num=10,
        retriever_penalty=1.2,
        reader_penalty=1,
        not_cross=False,
        not_mut=False,
        not_sort=False
    ):
        self.transformation = transformation
        self.pct_words_to_swap = pct_words_to_swap

        self.constraints = []
        self.pre_transformation_constraints = []
        for constraint in constraints:
            if isinstance(constraint, PreTransformationConstraint):
                self.pre_transformation_constraints.append(constraint)
            else:
                self.constraints.append(constraint)
    
        self.goal_function = goal_function
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.parents_num = parents_num
        self.retriever_penalty = retriever_penalty
        self.reader_penalty = reader_penalty

        self.survival = CustomSurvival()
        self.selection = CustomTournament(func_comp=binary_tournament)
        self.nds = NonDominatedSorting(method="fast_non_dominated_sort")

        self.not_cross = not_cross
        self.not_mut = not_mut
        self.not_sort = not_sort
        # internal flag to indicate if search should end immediately

    def _crossover_operation(self, pop_member1: Population, pop_member2: Population):

        pop1_words, pop1_indices = pop_member1.get_modified()
        pop2_words, pop2_indices = pop_member2.get_modified()

        maintain_indices = list(set(pop1_indices) & set(pop2_indices))

        pop1_only_indices = list(set(pop1_indices) - set(maintain_indices))
        pop2_only_indices = list(set(pop2_indices) - set(maintain_indices))

        cross_num = int((len(pop1_indices)- len(maintain_indices))*self.crossover_prob)
        maintain_num = len(pop1_indices)- len(maintain_indices) - cross_num
        if len(pop2_only_indices) >= cross_num and len(pop1_only_indices) >= maintain_num:
            maintain_indices += random.sample(pop1_only_indices, k=maintain_num)
            maintain_words = [pop1_words[pop1_indices.index(indice)] for indice in maintain_indices]
            
            cross_indices = random.sample(pop2_only_indices, k=cross_num)
            cross_words = [pop2_words[pop2_indices.index(indice)] for indice in cross_indices]

            assert not set(cross_indices) & set(maintain_indices)
            child_indices = maintain_indices + cross_indices
            child_word = maintain_words + cross_words
            return child_word, child_indices
        else:
            return [], []
        

    def _generate_population(self, per_words, per_words_indices):
        populations = []

        for w, i in zip(per_words, per_words_indices):
            attacked_instance = copy.deepcopy(self.current_text)
            attacked_instance.set_modified(w,i)
            populations.append(attacked_instance)

        attacked_texts = [population.get_perturbed_text() for population in populations]
        # og_scores = self.goal_function.eval([context], self.question, self.answers[0])
        scores = self.goal_function.eval(attacked_texts, self.question, self.answers[0])
        populations = [population.set_scores(score) for score, population in zip(scores, populations)]
        return populations

    def _mutation(self, words, indices):
        maintain_num = int(len(indices) * (1-self.mutation_prob))
        if maintain_num > 0:
            maintain_indices = random.choices(indices, k=maintain_num)
            maintain_words = [words[indices.index(indice)] for indice in maintain_indices]
            modified_indices = list(set(self.indices_to_modify) - set(maintain_indices))
            num_words_to_swap = int(self.pct_words_to_swap * len(self.indices_to_modify) - len(maintain_words))
            per_words, per_words_indices = self.transformation.get_perturbed_sequences(self.current_text.attacked_text, modified_indices, num_words_to_swap, 1)
            per_words = [maintain_words + pw for pw in per_words]
            per_words_indices = [maintain_indices + pwi for pwi in per_words_indices]
            populations = self._generate_population(per_words, per_words_indices)
            return populations
        else:
            return []


    def _crossover(self, pop_member1, pop_member2):
        results = []
        child_words, child_indices = self._crossover_operation(pop_member1, pop_member2)
        if child_words != []:
            if self.not_mut:
                # from IPython import embed; embed(); exit(0)
                results.extend(self._generate_population([child_words], [child_indices]))
            else:
                results.extend(self._mutation(child_words, child_indices))
        child_words, child_indices = self._crossover_operation(pop_member2, pop_member1)
        if child_words != []:
            if self.not_mut:
                results.extend(self._generate_population([child_words], [child_indices]))
            else:
                results.extend(self._mutation(child_words, child_indices))
        return results

    def _initialize_population(self):
        num_words_to_swap = max(
            int(self.pct_words_to_swap * len(self.indices_to_modify)), 1
        )
        per_words, per_words_indices = self.transformation.get_perturbed_sequences(self.current_text.attacked_text, self.indices_to_modify, num_words_to_swap, self.pop_size)
        populations = self._generate_population(per_words, per_words_indices)
        return populations

    def _get_modified_indices(self):
        indices_to_modify = set(range(len(self.current_text.attacked_text.words)))

        for constraint in self.pre_transformation_constraints:
            indices_to_modify = indices_to_modify & constraint(self.current_text.attacked_text, self.transformation)
        return indices_to_modify

    def attack_dataset(self, dataset):
        success = 0
        fail = 0
        results = []
        total_start_time = time.time()
        times = []
        for j in range(0, 5):
            for i, d in enumerate(tqdm(dataset)):

                start_time = time.time()
                answers = d["answers"]
                question = d["question"]
                ctxs = d["ctxs"]
                ctxs = ctxs[j:j+1]
                print(ctxs)
                # q_id = i
                q_id = d["id"]
                texts = [ctx["context"] for ctx in ctxs]
                # texts = [ctx["context"] for ctx in ctxs]
                gold_preds = self.goal_function.generate(texts, question)
                try:
                    for gold_pred, ctx in zip(gold_preds, ctxs):
                        # here I change the > to >= in case that some skip when format doesn't match
                        if EM(answers, gold_pred) >= 0:

                            doc_id = ctx["id"]
                            populations = self.perform_search(
                                ctx["context"],
                                question,
                                [gold_pred] + answers
                            )

                            # check = [r for r in populations[-1] if r[1][0] < 1]
                            # if len(check) > 0:
                                # check = sorted(check, key=lambda x: x[1][1])
                            pred = self.goal_function.generate(populations[-1][0][0], self.question)
                            if populations[-1][0][1][0] < 1:
                                em = EM(answers, pred[0])

                                if em == 0:
                                    logger.info(populations[-1][0])
                                    logger.info("Answer : {}, Pred: {}".format(answers, pred))
                                    success += 1
                                else:
                                    fail += 1

                                results.append({
                                    "q_id": q_id,
                                    "doc_id": doc_id,
                                    "question": question,
                                    "answers": answers,
                                    "ctx": ctx["context"],
                                    "att": populations,
                                    "og_pred": gold_pred,
                                    "att_pred": pred,
                                    "attribute": "success"
                                })
                            else:
                                fail += 1
                                results.append({
                                    "q_id": q_id,
                                    "doc_id": doc_id,
                                    "question": question,
                                    "answers": answers,
                                    "ctx": ctx["context"],
                                    "att": populations,
                                    "og_pred": gold_pred,
                                    "att_pred": pred,
                                    "attribute": "failure"
                                })
                            # the break will make an only result
                            # break
                        else:
                            logger.info("here! exit!")
                    if len(results) % 100 == 0 and len(results) > 0:
                        logger.info("S : {}, F : {}".format(success, fail))              
                    # if len(results) >= 100:
                    #     break
                except ZeroDivisionError:
                    pass

                end_time = time.time()
                elapsed_time = end_time - start_time
                times.append(elapsed_time)
                logger.info(f"Time taken: {elapsed_time:.4f} seconds")
                output_dir = "output_data"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                with open(os.path.join(output_dir, "{}.json".format(f"attacked-data-{j+1}")), 'w') as f: json.dump(results,f)
                with open(os.path.join(output_dir, "attacked-data-times.json"), 'w') as f:
                    json.dump(times, f)
        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time
        logger.info(f"Total time taken: {total_elapsed_time:.4f} seconds")
        return results

    def perform_search(self, context, question, answers):

        for C in self.pre_transformation_constraints:
            if isinstance(C, LabelConstraint):
                C.set_labels([AttackedText(answer) for answer in answers])

        self.context, self.question, self.answers = context, question, answers
        self.current_text = Population(context)
        og_scores = self.goal_function.eval([context], self.question, self.answers[0])
        self.current_text.og_scores = og_scores[0]
        self.indices_to_modify = self._get_modified_indices()

        populations = self._initialize_population()
        F = np.array([population.get_scores() for population in populations])
        fronts = self.nds.do(F, n_stop_if_ranked=100)
        populations = self.survival.do(F, populations, fronts, n_survive=self.pop_size)


        results = []
        for i in tqdm(range(self.max_iters)):

            if self.not_cross and self.not_mut:
                populations += self._initialize_population()
            else:
                if self.not_sort:
                    parents = [random.choices([i for i in range(len(populations))], k=2) for _ in range(self.parents_num)]
                else:
                    parents = self.selection.do(populations, self.parents_num, 2)
                children = []

                for p1, p2 in parents:
                    if self.not_cross:
                        pop_words, pop_indices = populations[p1].get_modified()
                        child = self._mutation(pop_words, pop_indices)
                        children.extend(child)
                        pop_words, pop_indices = populations[p2].get_modified()
                        child = self._mutation(pop_words, pop_indices)
                        children.extend(child)
                    else:
                        child = self._crossover(
                            populations[p1],
                            populations[p2],
                        )
                        children.extend(child)
                populations += children
            
            F = np.array([population.get_scores() for population in populations])

            for f in F:
                if f[0] > 1:
                    f[0] = f[0] * self.retriever_penalty
                if f[1] > 1:
                    f[1] = f[1] * self.reader_penalty
            fronts = self.nds.do(F, n_stop_if_ranked=self.pop_size)
            populations = self.survival.do(F, populations, fronts, n_survive=self.pop_size)
            result = [(population.get_perturbed_text(), population.get_scores(), population.get_metrics(), len(self.indices_to_modify)) for population in populations]
            results.append(result)

            if result[0][1][0] < 1:
                pred = self.goal_function.generate(result[0][0], self.question)
                em = EM(answers, pred[0])
                if em == 0:
                    logger.info(f"iteration: {i}")
                    return results
        logger.info(f"iteration: {i}")
        return results

