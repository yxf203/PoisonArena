import argparse
import os
from src.util import timestr, is_main

class Options:
    def __init__(self, task):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.task = task
        self.initialize()

    def initialize(self):
        # Basic Parameter
        self.parser.add_argument("--name", type=str, default="test")
        self.parser.add_argument("--dataset", type=str, default="nq")
        self.parser.add_argument("--split", type=str, default="test")
        self.parser.add_argument("--reader", type=str, default="Llama-8b")
        self.parser.add_argument("--retriever", type=str, default="contriever")

        self.parser.add_argument("--task", type=str, default="ODQA")

        # Directory
        self.parser.add_argument("--output_dir", type=str, default="./output/{name}/{t}")
        self.parser.add_argument("--data_dir", type=str, default="./data/{task}/{retriever}/{dataset}-{split}_100.json")
        self.parser.add_argument("--corpus_dir", type=str, default="./data/psgs_w100.tsv")
        self.parser.add_argument("--template_dir", type=str, default="./data/template/{reader}.json")
        self.parser.add_argument("--model_dir", type=str, default="change to your model path")

        # GPU Setting
        self.parser.add_argument("--gpus", type=int, default=2)
        self.parser.add_argument("--num_workers", type=int, default=5)
        self.parser.add_argument("--batch_size", type=int, default=16)

        self.parser.add_argument("--passage_maxlength", type=int, default=512)
        self.parser.add_argument("--max_new_tokens", type=int, default=30)

        self.parser.add_argument("--is_save", action="store_true")
        self.parser.add_argument("--is_vllm", action="store_true")
            

        if self.task == "attack":

            # Common Hyperparameter
            self.parser.add_argument("--is_black", action="store_true")
            self.parser.add_argument("--is_genetic", action="store_true")
            self.parser.add_argument("--is_hotflip", action="store_true")
            self.parser.add_argument('--method', type=str, default="typo")

            # White Box Attack Hyperparameter
            self.parser.add_argument("--threshold", type=float, default=0.95)
            self.parser.add_argument("--with_wordswap", action="store_true")

            # Black Box Attack Hyperparameter
            self.parser.add_argument("--max_iters", type=int, default=30)
            self.parser.add_argument("--per_prob", type=float, default=0.2)
            self.parser.add_argument("--perturbation_level", type=float, default=0.2)
            self.parser.add_argument("--transformations_per_example", type=int, default=100)
            self.parser.add_argument("--crossover_prob", type=float, default=0.2)
            self.parser.add_argument("--mutation_prob", type=float, default=0.4)
            self.parser.add_argument("--parents_num", type=int, default=10)
            self.parser.add_argument("--retriever_penalty", type=float, default=1.1)
            self.parser.add_argument("--reader_penalty", type=float, default=1.0)

            self.parser.add_argument("--not_cross", action="store_true")
            self.parser.add_argument("--not_mut", action="store_true")
            self.parser.add_argument("--not_sort", action="store_true")
        else:
            raise NotImplementedError("Not supported task.")

    def print_options(self, opt):
        message = "\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = f"\t[default: %s]" % str(default)
            message += f"{str(k):>40}: {str(v):<40}{comment}\n"
        return message

    def parse(self, t):
        opt, _ = self.parser.parse_known_args()

        opt.output_dir = opt.output_dir.format(name=opt.name, t=t)
        opt.data_dir = opt.data_dir.format(task=opt.task, retriever=opt.retriever, dataset=opt.dataset, split=opt.split)
        opt.template_dir = opt.template_dir.format(reader=opt.reader.split("-")[0])

        os.makedirs(opt.output_dir, exist_ok=True)
        print(opt.data_dir)   
        assert(os.path.exists(opt.data_dir))
        assert(os.path.exists(opt.template_dir))
        message = self.print_options(opt)
        return opt, message
