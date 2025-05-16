import numpy as np
import torch

from textattack.constraints import Constraint
from textattack.constraints.semantics.sentence_encoders import SentenceEncoder
from textattack.constraints import PreTransformationConstraint
from src.retriever import Retriever
from transformers import AutoTokenizer

from src.util import get_sim

class Retrieve_Constraint(Constraint):
    """An abstract class that represents constraints on adversial text
    examples. Constraints evaluate whether transformations from a
    ``AttackedText`` to another ``AttackedText`` meet certain conditions.

    Args:
        compare_against_original (bool): If `True`, the reference text should be the original text under attack.
            If `False`, the reference text is the most recent text from which the transformed text was generated.
            All constraints must have this attribute.
    """

    def __init__(self, opt, **kwargs):
        super().__init__(**kwargs)
        self.model = Retriever(opt)
        self.threshold = opt.threshold
        self.tokenizer = self.model.get_tokenizer()
        self.batch_size = opt.batch_size
        self.tokenizer_kwargs = {
            "max_length":opt.passage_maxlength,
            "truncation":True,
            "padding":True, 
            "return_tensors":"pt"
        }
    
    def call_many(self, transformed_texts, reference_text, query):
        """Filters ``transformed_texts`` based on which transformations fulfill
        the constraint. First checks compatibility with latest
        ``Transformation``, then calls ``_check_constraint_many``

        Args:
            transformed_texts (list[AttackedText]): The candidate transformed ``AttackedText``'s.
            reference_text (AttackedText): The ``AttackedText`` to compare against.
        """
        incompatible_transformed_texts = []
        compatible_transformed_texts = []
        for transformed_text in transformed_texts:
            try:
                if self.check_compatibility(
                    transformed_text.attack_attrs["last_transformation"]
                ):
                    compatible_transformed_texts.append(transformed_text)
                else:
                    incompatible_transformed_texts.append(transformed_text)
            except KeyError:
                raise KeyError(
                    "transformed_text must have `last_transformation` attack_attr to apply constraint"
                )
        filtered_texts = self._check_constraint_many(
            compatible_transformed_texts, reference_text, query
        )
        return list(filtered_texts) + incompatible_transformed_texts

    def _score_list(self, reference_text, transformed_texts, query):
        results = []
        ctxs = reference_text.text
        ctxs_embedding = self.tokenizer(
            ctxs, **self.tokenizer_kwargs
        )
        query_embedding = self.tokenizer(query, **self.tokenizer_kwargs)
        og_scores = self.model.forward(query_embedding, ctxs_embedding)
        for i in range(0, len(transformed_texts), self.batch_size):
            a_ctxs = [text.text for text in transformed_texts[i:min(i+self.batch_size, len(transformed_texts))]]
            qs = [query for _ in range(len(a_ctxs))]

            a_ctxs_embedding = self.tokenizer(
                a_ctxs, **self.tokenizer_kwargs
            )
    
            qs_embedding = self.tokenizer(
                qs, **self.tokenizer_kwargs
            )

            at_scores = self.model.forward(qs_embedding, a_ctxs_embedding)
            results += [a > og_scores[0] * self.threshold for a in at_scores]    
        return torch.tensor(results)

    def _check_constraint_many(self, transformed_texts, reference_text, query):
        """Filters the list ``transformed_texts`` so that the similarity
        between the ``reference_text`` and the transformed text is greater than
        the ``self.threshold``."""
        mask = self._score_list(reference_text, transformed_texts, query)
        if len(mask) > 1:
            return np.array(transformed_texts)[mask]
        else:
            if mask[0]:
                return transformed_texts
            else:
                return []

    def _check_constraint(self, transformed_text, reference_text, query):
        mask = self._score_list(reference_text, [transformed_text], query)
        return mask[0]

    def extra_repr_keys(self):
        return [
            "threshold"
        ] + super().extra_repr_keys()
    
class LabelConstraint(PreTransformationConstraint):
    """
    A constraint that does not allow to attack the labels (or any words that is important for tasks) in the prompt.
    """

    def __init__(self, labels=[]):
        self.set_labels(labels)

    def set_labels(self, labels=[]):
        self.labels = []
        for label in labels:
            for word in label.words:
                self.labels.append(str(word).lower())
        # self.labels = [label.lower() for label in labels]

    def _get_modifiable_indices(self, current_text):
        modifiable_indices = set()
        modifiable_words = []
        for i, word in enumerate(current_text.words):
            if str(word).lower() not in self.labels:
                modifiable_words.append(word)
                modifiable_indices.add(i)
        return modifiable_indices

    def check_compatibility(self, transformation):
        """
        It is always true.
        """
        return True