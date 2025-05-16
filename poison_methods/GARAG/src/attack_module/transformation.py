import copy
import random
import re
import itertools
from tqdm import tqdm
import numpy as np
import os

import numpy as np
from textattack.transformations.word_swaps import WordSwap
from itertools import combinations, product


class SimpleAttack(WordSwap):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.word_dict = {}

    def __call__(
        self,
        current_text,
        pre_transformation_constraints=[],
        indices_to_modify=None,
        shifted_idxs=False,
        return_indices=False,
        perturb_words = -1,
        return_seqs = -1
    ):
        if indices_to_modify is None:
            indices_to_modify = set(range(len(current_text.words)))
            # If we are modifying all indices, we don't care if some of the indices might have been shifted.
            shifted_idxs = False
        else:
            indices_to_modify = set(indices_to_modify)

        if shifted_idxs:
            indices_to_modify = set(
                current_text.convert_from_original_idxs(indices_to_modify)
            )

        for constraint in pre_transformation_constraints:
            indices_to_modify = indices_to_modify & constraint(current_text, self)
            
        if return_indices:
            return indices_to_modify

        transformed_texts = self._get_transformations(current_text, indices_to_modify, perturb_words=perturb_words, return_seqs=return_seqs)
        for text in transformed_texts:
            text.attack_attrs["last_transformation"] = self
        return transformed_texts

    def _get_replacement_words(self, word):
        raise NotImplementedError()

    def get_perturbed_sequences(self, current_text, indices_to_modify, perturb_words, return_seqs):
        transformed_texts = 0
        tries = 0
        perturb_words = min(len(indices_to_modify), perturb_words)
        words = current_text.words
        per_words = []
        per_words_indices = []
        try:
            while transformed_texts < return_seqs or tries < 3*return_seqs:

                per_indices = random.sample(list(indices_to_modify), k=perturb_words)
                per_word = []
                per_words_indice = []

                for ii, indice in enumerate(per_indices):

                    replacement_words = []
                    word_to_replace = words[indice]

                    if word_to_replace in self.word_dict:
                        replacement_words += self.word_dict[word_to_replace]

                    if len(replacement_words) <= 5:
                        replacement_words += list(set(self._get_replacement_words(word_to_replace)))
                        self.word_dict[word_to_replace] = replacement_words
                        
                    if replacement_words != []:
                        r = random.choices(replacement_words)
                        per_word.extend(r)
                        per_words_indice.append(indice)

                if per_word != []:
                    per_words.append(per_word)
                    per_words_indices.append(per_words_indice)
                    transformed_texts += 1
                tries += 1

        except KeyboardInterrupt:
            from IPython import embed; embed(); exit(0)

        return per_words, per_words_indices

    def get_perturbed_sequences_forgreedy(self, current_text, indices_to_modify, return_seqs):
        transformed_texts = 0
        tries = 0

        words = current_text.words
        per_words = []
        per_words_indices = []

        while transformed_texts < return_seqs or tries < 3*return_seqs:

            per_word = []
            per_words_indice = []

            for ii, indice in enumerate(indices_to_modify):

                replacement_words = []
                word_to_replace = words[indice]

                if word_to_replace in self.word_dict:
                    replacement_words += self.word_dict[word_to_replace]

                if len(replacement_words) <= 5:
                    replacement_words += list(set(self._get_replacement_words(word_to_replace)))
                    self.word_dict[word_to_replace] = replacement_words
                    
                if replacement_words != []:
                    r = random.choices(replacement_words)
                    per_word.extend(r)
                    per_words_indice.append(indice)

            if per_word != []:
                per_words.append(per_word)
                per_words_indices.append(per_words_indice)
                transformed_texts += 1
            tries += 1
            
        return per_words, per_words_indices

    def _get_transformations(self, current_text, indices_to_modify, perturb_words, return_seqs):
        transformed_texts = []
        if perturb_words > 0:
            transformed_texts_idx = []
            perturb_words = min(len(indices_to_modify), perturb_words)
            while len(transformed_texts) < return_seqs:
                per_indices = random.sample(list(indices_to_modify), k=perturb_words)
                per_words = []
                per_words_indices = []
                # per_indices = random.shuffle(list(combinations(list(indices_to_modify, perturb_words))))
                attacked_text = current_text
                words = current_text.words
                for ii, indice in enumerate(per_indices):
                    replacement_words = []
                    word_to_replace = words[indice]

                    if word_to_replace in self.word_dict:
                        replacement_words += self.word_dict[word_to_replace]

                    if len(replacement_words) <= 5:
                        replacement_words += list(set(self._get_replacement_words(word_to_replace)))
                        self.word_dict[word_to_replace] = replacement_words
                        
                    if replacement_words != []:
                        prob = [(1-self.per_prob)/len(replacement_words)] * len(replacement_words) + [self.per_prob]
                        r = random.choices(replacement_words + [word_to_replace], weights=prob)
                        per_words_indices.append(indice)
                        per_words.append(r)
                        
                attacked_text = attacked_text.replace_words_at_indices(per_words_indices, per_words)
                transformed_texts.append(attacked_text)
            return transformed_texts
        
        elif perturb_words == -1:
            for i in indices_to_modify:
                word_to_replace = words[i]
                replacement_words = self._get_replacement_words(word_to_replace)
                transformed_texts_idx = []
                for r in replacement_words:
                    if r == word_to_replace:
                        continue
                    transformed_texts_idx.append(current_text.replace_word_at_index(i, r))
                transformed_texts.extend(transformed_texts_idx)
            return transformed_texts
        
        else:
            NotImplementedError()

class Typo(SimpleAttack):
    def __init__(self):
        super().__init__()
        self.typos = {}
        for line in open(os.path.join("./data/noise", "en.natural")):
            line = line.strip().split()
            self.typos[line[0]] = line[1:]
        # Keyboard Typo
        self.NN = {}
        for line in open(os.path.join("./data/noise", "en.key")):
            line = line.split()
            self.NN[line[0]] = line[1:]
        # self.anthro = ANTHRO()
        # self.anthro.load("./data/noise/ANTHRO_Data_V1.0")
    
    def _get_replacement_words(self, word):
        per_words = self._get_keyboard_typo(word) + self._get_natural_typo(word) + self._get_truncate(word) + self._get_innerswap(word)
        return per_words

    def _get_keyboard_typo(self, word):
        words = []
        word = list(word)
        for i in range(len(word)):
            char = word[i]
            if char in self.NN:
                for w in self.NN[char.lower()]:
                    replace_word = copy.deepcopy(word)
                    replace_word[i] = w
                    words.append(''.join(replace_word))
            elif char.lower() in self.NN:
                for w in self.NN[char.lower()]:
                    replace_word = copy.deepcopy(word)
                    replace_word[i] = w.upper()
                    words.append(''.join(replace_word))
        return words
    
    def _get_natural_typo(self, word):
        if word in self.typos:
            return self.typos[word]
        return []

    def _get_truncate(self, word: str, minlen: int = 3, cutoff: int = 3):
        """
        TODO: docs
        :param cutoff:
        :param minlen:
        :param word:
        :return:
        """
        words =[]

        chars = list(word)
        tmp_cutoff = cutoff
        while len(chars) > minlen and tmp_cutoff > 0:
            chars = chars[:-1]
            tmp_cutoff -= 1
            words.append(''.join(chars))
        chars = list(word)
        tmp_cutoff = cutoff
        while len(chars) > minlen and tmp_cutoff > 0:
            chars = chars[1:]
            tmp_cutoff -= 1
            words.append(''.join(chars))
        return words

    def _get_innerswap(self, word: str):
        def __shuffle_string__(_word: str, _seed=42):
            """
            shuffles the given string if a seed is given it shuffles in respect to the given seed.

            hello world -> elloh roldw

            :param _seed: seed
            :param _word: string (word) to shuffle
            :return: shuffled string
            """
            chars = list(_word)
            if _seed is not None:
                np.random.seed(_seed)
            np.random.shuffle(chars)
            return ''.join(chars)

        words = []
        if len(word) <= 3:
            return words
        tries = 0
        min_perturb = min(int(len(word)*0.4),2)
        while tries < 5:
            tries += 1  # we can get a deadlock if the word is e.g. maas
            start = random.randrange(1,len(word)-min_perturb+1)
            first, mid, last = word[0:start], word[start:start+min_perturb], word[start+min_perturb:]
            words.append(first + __shuffle_string__(mid) + ''.join(last))
        words = list(set(words))
        return words

    def _get_anthro_typo(self, word):
        per_words = list(self.anthro.get_similars(word, level=1, distance=1, strict=True))
        return per_words

class Keyboard(Typo):

    def _get_replacement_words(self, word):
        per_words = self._get_keyboard_typo(word)
        return per_words

class Natural(Typo):

    def _get_replacement_words(self, word):
        per_words = self._get_natural_typo(word)
        return per_words

class Truncate(Typo):

    def _get_replacement_words(self, word):
        per_words = self._get_truncate(word)
        return per_words

class InnerSwap(Typo):

    def _get_replacement_words(self, word):
        per_words = self._get_innerswap(word)
        return per_words
