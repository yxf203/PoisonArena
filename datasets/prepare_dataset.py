from beir import util
import os

# Download and save dataset
# datasets = ['nq', 
            # 'msmarco']
datasets = ['nq']
for dataset in datasets:
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.getcwd()
    corpus_path = os.path.join(out_dir, dataset, 'corpus.jsonl')
    data_path = os.path.join(out_dir, dataset)
    if not os.path.exists(corpus_path):
        data_path = util.download_and_unzip(url, out_dir)

os.system('rm *.zip')