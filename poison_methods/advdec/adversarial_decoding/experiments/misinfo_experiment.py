import json
import faiss, random, gc, torch
from sentence_transformers import SentenceTransformer
from adversarial_decoding.strategies.retrieval_decoding import RetrievalDecoding
from adversarial_decoding.utils.utils import append_to_target_dir, file_device
import time
def misinfo_experiment(args):
    print("enter misinfo experiment")
    device = file_device
    encoder = SentenceTransformer(args.retriever_path, trust_remote_code=True, device=device, model_kwargs={'torch_dtype': torch.bfloat16})

    short_file_name = args.attack_data_name

    file_name = f'{args.attack_data_dir}/{short_file_name}'

    print(f"now is processing {file_name}")
    with open(f'./datasets/{file_name}.json', 'r') as f:
        misinfo_dict = json.load(f)
    target_dir = f'./data/{file_name}_misinfo_contriever_reverse_dot.json'
    iter_times = []
    for p in misinfo_dict:
        start_time = time.time()
        id, index = p["id"], p["index_cluster"]
        misinfo, target_queries = p['misinfo'], p['queries']
        random.seed(42)
        random.shuffle(target_queries)
        # train_queries, test_queries = target_queries[:20], target_queries[20:]
        train_queries, test_queries = target_queries[:5], target_queries[5:]
        print("train_queries:", train_queries)
        prompt_texts = []
        results = []
        adv_texts = []
        # cos_sim_s = []
        dot_sim_s = []
        naturalness_s = []
        for m in misinfo:
            # prompt_text = f'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{misinfo}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            prompt_text = f'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{m}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            # control_text = misinfo
            control_text = m
            target_text = ''
            # target_text = 'Anny John'
            cluster_candidates = []
            for cluster_label in range(1):
                should_natural = True
                attack = RetrievalDecoding('', control_text, cluster_label, 3, encoder, train_queries=train_queries, device=file_device, should_natural=should_natural, model_path=args.model_path)
                attack.get_combined_scorer(prompt_text, target_text)
                beam_width=10
                max_steps=30
                top_k=10
                top_p=0.999
                best_candidate = attack.run_decoding(
                    prompt=prompt_text,
                    target=target_text,
                    beam_width=beam_width,
                    max_steps=max_steps,
                    top_k=top_k,
                    top_p=top_p,
                    should_full_sent=False
                )
                del attack
                cluster_candidates.append(best_candidate)
            result = [cand.seq_str for cand in cluster_candidates]
            # cos_sim = [cand.cos_sim for cand in cluster_candidates]
            dot_sim = [cand.dot_sim for cand in cluster_candidates]
            naturalness = [cand.naturalness for cand in cluster_candidates]
            gc.collect()
            torch.cuda.empty_cache()
            parameters = {
                'beam_width': beam_width,
                'max_steps': max_steps,
                'top_k': top_k,
                'top_p': top_p,
            }
            prompt_texts.append(prompt_text)
            results.append(result)
            # adv_texts.append(result[0] + control_text)
            # cos_sim_s.append(cos_sim[0])
            dot_sim_s.append(dot_sim[0])
            naturalness_s.append(naturalness[0])
            adv_texts.append(control_text + result[0])
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Peak GPU memory usage: {peak_memory:.2f} GB")
        # append_to_target_dir(target_dir, parameters | {'prompt': prompt_text, 'target': target_text, 'result': result, 'cos_sim': cos_sim, 'naturalness': naturalness})
        append_to_target_dir(target_dir, parameters | {'id': id, 'index_cluster': index,'prompt': prompt_texts, 'target': target_text, 'result': results, "adv_texts": adv_texts, 'dot_sim': dot_sim_s, 'naturalness': naturalness_s})
        # append_to_target_dir(target_dir, parameters | {'id': id, 'index_cluster': index,'prompt': prompt_texts, 'target': target_text, 'result': results, "adv_texts": adv_texts, 'cos_sim': cos_sim_s, 'naturalness': naturalness_s})
        end_time = time.time()
        iter_times.append(end_time - start_time)
        print(f"time: {end_time - start_time:.2f}s") 
        avg_time = sum(iter_times) / len(iter_times)
        attack_time_data = {
            "average_time": avg_time,
            "peak_memory": peak_memory,
            "iter_times": iter_times,
        }
        output_path = f"./time/{short_file_name}_attack_time.json"
        with open(output_path, "w") as f:
            json.dump(attack_time_data, f, indent=4)
        # print(f"Attack times and average time written to {output_path}")
