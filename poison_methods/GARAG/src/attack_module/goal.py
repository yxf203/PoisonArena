import json
import numpy as np
from textattack.models.wrappers import ModelWrapper
from textattack.goal_functions import GoalFunction
from textattack.goal_function_results import GoalFunctionResultStatus, GoalFunctionResult
from textattack.shared.utils import add_indent
from src.reader import load_reader
from src.retriever import load_retriever
from src.util import EM, f1

import torch


class CustomGoalFunctionResult(GoalFunctionResult):

    def __init__(
        self,
        qusetion,
        answers,
        og_text,
        attacked_text,
        ground_truth_output,
        output,
        goal_status,
        score,
        num_queries,
        goal_function_result_type="",
    ):
        self.question = qusetion
        self.answers = answers
        self.og_text = og_text
        self.attacked_text = attacked_text
        self.og_output = ground_truth_output
        self.output = output
        self.score = score
        self.goal_status = goal_status
        self.num_queries = num_queries
        self.ground_truth_output = ground_truth_output
        self.goal_function_result_type = goal_function_result_type

        if isinstance(self.output, torch.Tensor):
            self.output = self.output.numpy()

        if isinstance(self.score, torch.Tensor):
            self.score = self.score.item()

    def __repr__(self):
        main_str = "GoalFunctionResult( "
        lines = []
        lines.append(
            add_indent(
                f"(goal_function_result_type): {self.goal_function_result_type}", 2
            )
        )
        lines.append(add_indent(f"(question): {self.question}", 2))
        lines.append(add_indent(f"(og_text): {self.og_text}", 2))
        lines.append(add_indent(f"(attacked_text): {self.attacked_text.text}", 2))
        lines.append(
            add_indent(f"(ground_truth_output): {self.ground_truth_output}", 2)
        )
        lines.append(add_indent(f"(model_output): {self.output}", 2))
        lines.append(add_indent(f"(score): {self.score}", 2))
        main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str

    def get_text_color_input(self):
        """A string representing the color this result's changed portion should
        be if it represents the original input."""
        return "red"

    def get_text_color_perturbed(self):
        """A string representing the color this result's changed portion should
        be if it represents the perturbed input."""
        return "blue"

    def get_colored_output(self, color_method=None):
        """Returns a string representation of this result's output, colored
        according to `color_method`."""
        return str(self.output)

class Reader_Wrapper(ModelWrapper):
    def __init__(self, opt, template):
        self.model = load_reader(opt)
        self.is_gpt = "chatgpt" in opt.reader
        if not self.is_gpt:
            self.is_vllm = opt.is_vllm
            self.tokenizer = self.model.get_tokenizer()
        self.template = template
        self.batch_size = opt.batch_size

    def __call__(self, attacked_contexts, question, answer):
        results = []
        inputs = [self.template.format(q=question, d=text) for text in attacked_contexts]
        labels = [answer] * len(inputs)
        if self.is_gpt:
            results += self.model.get_scores(attacked_contexts, question, answer)
        elif self.is_vllm:
            # inputs = [input + " " + label for input, label in zip(inputs, labels)]
            inputs = [input + label for input, label in zip(inputs, labels)]
            # print("labelsss:", labels)
            # print("inputs:",inputs)
            results += self.model.get_scores(inputs, labels)
        else:

            input_embeddings = self.tokenizer(
                inputs,
                max_length=512,
                truncation=True,
                padding=True, 
                return_tensors="pt",
            )
            label_embeddings = self.tokenizer(
                labels, 
                max_length=512,
                truncation=True,
                padding=True, 
                return_tensors="pt",
            )
            results += self.model.get_scores(input_embeddings.input_ids, label_embeddings.input_ids)

        return results
    
    def generate(self, contexts, question):

        inputs = [self.template.format(q=question, d=context) for context in contexts]
        if self.is_gpt:
            return self.model(contexts, question)
        if self.is_vllm:
            # print("template!!")
            # messages = [
            #     {"role": "system", "content": "You are a concise assistant. Provide only a very short phrase as an answer, such as '1998', 'May 16th, 1931', or 'James Bond'. Do not provide any explanations, clarifications, or extra content. Just answer the question, nothing more."},
            #     {"role": "user", "content": inputs[0]},
            # ]
            # input_ids = self.tokenizer.apply_chat_template(
            #     messages,
            #     add_generation_prompt=True,
            #     tokenize=False
            #     # return_tensors="pt"
            # )
            # print(input_ids)
            # print("done!")
            return self.model(inputs)
        else:
            inputs = self.tokenizer(
                    inputs,
                    max_length=512,
                    truncation=True,
                    padding=True, 
                    return_tensors="pt",
            )
            return self.model(**inputs)

class Retriever_Wrapper(ModelWrapper):
    def __init__(self, opt):
        self.model = load_retriever(opt)
        self.tokenizer = self.model.get_tokenizer()
        self.batch_size = opt.batch_size
        self.tokenizer_kwargs = {
            "max_length":opt.passage_maxlength,
            "truncation":True,
            "padding":True, 
            "return_tensors":"pt"
        }

    def __call__(self, question, contexts):
        results = []

        for i in range(0, len(contexts), self.batch_size):
            ctx = contexts[i:min(i+self.batch_size, len(contexts))]
            q = [question] * len(ctx)
    
            ctxs_embedding = self.tokenizer(
                ctx, **self.tokenizer_kwargs
            )
        
            qs_embedding = self.tokenizer(
                q, **self.tokenizer_kwargs
            )

            scores = [float(score) for score in self.model(qs_embedding, ctxs_embedding)]

            results += scores 

        return results
    
    def get_doc_emb(self, contexts):
        results = []
        for i in range(0, len(contexts), self.batch_size):
            ctx = contexts[i:min(i+self.batch_size, len(contexts))]
            ctxs_embedding = self.tokenizer(
                ctx, **self.tokenizer_kwargs
            )
            ctxs_embedding.to(self.model.d_encoder.device)
            context_embeddings = self.model.d_encoder(**ctxs_embedding)
            results.append(context_embeddings)
        return results
    
    def get_doc_emb_2(self, contexts):
        inputs = {

        }
        for k,v in contexts.items():
            inputs[k] = v.to(self.model.d_encoder.device)
        return self.model.d_encoder(**inputs)

    def get_doc_token(self, contexts):
        return self.tokenizer(
                contexts, **self.tokenizer_kwargs
            )
    
    def get_q_emb(self, question):
        q_embedding = self.tokenizer(
            [question], **self.tokenizer_kwargs
        )
        q_embedding.to(self.model.q_encoder.device)
        q_embeddings = self.model.q_encoder(**q_embedding)
        return q_embeddings

class Single_GoalFunction(GoalFunction):

    def init_attack_example(self, attacked_text, raw_data):
        """Called before attacking ``attacked_text`` to 'reset' the goal
        function and set properties for this example."""
        self.initial_attacked_text = attacked_text
        self.raw_data = raw_data
        self.num_queries = 0
        result, _ = self.get_result(attacked_text, check_skip=True)
        return result, _

    def get_result(self, attacked_text, **kwargs):
        """A helper method that queries ``self.get_results`` with a single
        ``AttackedText`` object."""
        results, search_over = self.get_results([attacked_text], **kwargs)
        result = results[0] if len(results) else None
        return result, search_over

    def _call_model_uncached(self, attacked_text_list):
        """Queries model and returns outputs for a list of AttackedText
        objects."""
        if not len(attacked_text_list):
            return []

        inputs = [at.text for at in attacked_text_list]
        outputs = []
        i = 0
        while i < len(inputs):
            batch = inputs[i : i + self.batch_size]
            batch_preds = self.model(batch, self.raw_data["question"])

            # Some seq-to-seq models will return a single string as a prediction
            # for a single-string list. Wrap these in a list.
            if isinstance(batch_preds, str):
                batch_preds = [batch_preds]

            # Get PyTorch tensors off of other devices.
            if isinstance(batch_preds, torch.Tensor):
                batch_preds = batch_preds.cpu()

            if isinstance(batch_preds, list):
                outputs.extend(batch_preds)
            elif isinstance(batch_preds, np.ndarray):
                # outputs.append(batch_preds)
                outputs.append(torch.tensor(batch_preds))
            else:
                outputs.append(batch_preds)
            i += self.batch_size

        if isinstance(outputs[0], torch.Tensor):
            outputs = torch.cat(outputs, dim=0)
        elif isinstance(outputs[0], np.ndarray):
            outputs = np.concatenate(outputs).ravel()

        assert len(inputs) == len(
            outputs
        ), f"Got {len(outputs)} outputs for {len(inputs)} inputs"

        return self._process_model_outputs(attacked_text_list, outputs)

    def _call_model(self, attacked_text_list):
        """Gets predictions for a list of ``AttackedText`` objects.

        Gets prediction from cache if possible. If prediction is not in
        the cache, queries model and stores prediction in cache.
        """
        if not self.use_cache:
            return self._call_model_uncached(attacked_text_list)
        else:
            uncached_list = []
            for text in attacked_text_list:
                if text in self._call_model_cache:
                    # Re-write value in cache. This moves the key to the top of the
                    # LRU cache and prevents the unlikely event that the text
                    # is overwritten when we store the inputs from `uncached_list`.
                    self._call_model_cache[text] = self._call_model_cache[text]
                else:
                    uncached_list.append(text)
            uncached_list = [
                text
                for text in attacked_text_list
                if text not in self._call_model_cache
            ]
            outputs = self._call_model_uncached(uncached_list)
            for text, output in zip(uncached_list, outputs):
                self._call_model_cache[text] = output
            all_outputs = [self._call_model_cache[text] for text in attacked_text_list]
            return all_outputs

    def get_results(self, attacked_text_list, check_skip=False):
        """For each attacked_text object in attacked_text_list, returns a
        result consisting of whether or not the goal has been achieved, the
        output for display purposes, and a score.

        Additionally returns whether the search is over due to the query
        budget.
        """
        results = []
        if self.query_budget < float("inf"):
            queries_left = self.query_budget - self.num_queries
            attacked_text_list = attacked_text_list[:queries_left]
        self.num_queries += len(attacked_text_list)
        if isinstance(self.model, Reader_Wrapper):
            model_outputs = self._call_model(attacked_text_list)
        else:
            NotImplementedError()
        for attacked_text, model_output in zip(attacked_text_list, model_outputs):
            displayed_output = self._get_displayed_output(model_output)
            # from IPython import embed; embed()
            goal_status = self._get_goal_status(
                model_output, attacked_text, check_skip=check_skip
            )
            goal_function_score = self._get_score(attacked_text, self.raw_data["pred"])
            results.append(
                self._goal_function_result_type()(
                    self.raw_data["question"],
                    self.raw_data["answers"],
                    self.raw_data["context"],
                    attacked_text,
                    self.raw_data["pred"],
                    model_output,
                    goal_status,
                    goal_function_score,
                    self.num_queries,
                )
            )
            # from IPython import embed; embed(); exit(0)
        return results, self.num_queries == self.query_budget

    def _process_model_outputs(self, inputs, outputs):
        return outputs

    def _get_goal_status(self, model_output, attacked_text, check_skip=False):
        should_skip = check_skip and self._should_skip(model_output, attacked_text)
        # from IPython import embed; embed()
        if should_skip:
            return GoalFunctionResultStatus.SKIPPED
        if self.maximizable:
            return GoalFunctionResultStatus.MAXIMIZING
        if self._is_goal_complete(model_output, attacked_text):
            return GoalFunctionResultStatus.SUCCEEDED
        return GoalFunctionResultStatus.SEARCHING

    def _is_goal_complete(self, model_output, attacked_text):
        answers, pred = self.raw_data["answers"], self.raw_data["pred"]
        if f1(answers, model_output) < f1(answers, pred):
            return True
        else:
            return False
            
    def _get_score(self, attacked_text, pred):
        score = self.model._get_scores([attacked_text], self.raw_data["question"], pred)
        return 1-score[0]
    
    def _goal_function_result_type(self):
        """Returns the class of this goal function's results."""
        return CustomGoalFunctionResult

class Double_GoalFunction:

    def __init__(self, opt):
        with open(opt.template_dir, 'r') as f: 
            template = json.load(f)[0]
        self.reader = Reader_Wrapper(opt, template)
        self.retriever = Retriever_Wrapper(opt)

    def eval(self, contexts, questions, answers):
        with torch.no_grad():
            retriever_results = self.retriever(questions, contexts)
            reader_results = self.reader(contexts, questions, answers)
            results = [[r1,r2] for r1,r2 in zip(retriever_results, reader_results)]
        return results
    
    def generate(self, contexts, question):
        if isinstance(contexts, list):
            with torch.no_grad():
                return self.reader.generate(contexts, question)
        else:
            with torch.no_grad():
                return self.reader.generate([contexts], question)
