# 🥊PoisonArena: Uncovering Competing Poisoning Attacks in Retrieval-Augmented Generation

Here is the codebase for **PoisonArena**🏟️ — introduced in our paper as a novel benchmark designed to evaluate competing data poisoning attacks in **Retrieval-Augmented Generation (RAG)** systems under **multi-adversary** settings.

This codebase also includes re-implementations of several state-of-the-art poisoning attack methods.

## 📚 Included Attack Methods

Thanks to the following amazing works that we integrated:

-  [**PoisonedRAG**](https://github.com/sleeepeer/PoisonedRAG)
-  [**AdvDec**](https://github.com/collinzrj/adversarial_decoding)
-  [**GARAG**](https://github.com/zomss/GARAG)
-  [**GASLITE**](https://github.com/matanbt/gaslite)
-  [**Corpus-Poison**](https://github.com/princeton-nlp/corpus-poisoning)
-  [**Content-Poisoning**](https://github.com/ZQ-Struggle/Content-Poisoning)

### 🛡️ Defense Method

- 🧷 [**InstructRAG**](https://github.com/weizhepei/InstructRAG)

## 🗂️ Repository Structure

```bash
├── beir_results/           # Precomputed BEIR results for building top-k indices
├── datasets/
├── combat/                 # Our main experiments
│   ├── data/               # Poisoned and original datasets
├── poison_methods/         # All poisoning methods integrated into PoisonArena
│   ├── poisonedrag/
│   ├── advdec/
│   ├── garag/
│   ├── gaslite/
│   ├── corpus_poison/
│   └── content_poison/
├── model_configs/          # Model paths and parameter configurations
├── models/                 # LLM checkpoints (e.g. LLaMA3-8B-Instruct)
├── retriever/              # Retriever models (e.g. Contriever)
└── README.md
```

## 🔗Quick Link

- [🔧Prerequisites](#--prerequisites)
  * [🔹1. Prepare the Data](#--1-prepare-the-data)
  * [🔹2. Prepare Models](#--2-prepare-models)
  * [🔹3. Evaluate BEIR](#--3-evaluate-beir)
  * [🔹4. Prepare Arena Data](#--4-prepare-arena-data)
- [🏟️ PoisonArena](#----poisonarena)
  * [📁 Project Structure](#---project-structure)
  * [⚔️ How to Run PoisonArena](#---how-to-run-poisonarena)
- [💻 Reproducing Experiments](#---reproducing-experiments)
  * [🔹 1. COMBAT](#---1-combat)
  * [🔹 2. Poison Methods](#---2-poison-methods)
    + [2.1 PoisonedRAG](#21-poisonedrag)
    + [2.2 AdvDec](#22-advdec)
    + [2.3 GARAG](#23-garag)
    + [2.4 GASLITE](#24-gaslite)
    + [2.5 Corpus-Poison](#25-corpus-poison)
    + [2.6 Content-Poison](#26-content-poison)
- [🙏 Acknowledgements](#---acknowledgements)

## 🔧Prerequisites

### 🔹1. Prepare the Data

We use the [BEIR datasets](https://github.com/beir-cellar/beir) in our experiments. Before starting the attack process, please ensure you have the BEIR results prepared. The most important file is `corpus.jsonl`, as it simulates the knowledge base. You can also use your own corpus as the knowledge base if preferred.

To prepare the dataset, run:

```bash
cd datasets
python prepare_dataset.py
```

### 🔹2. Prepare Models

Please place your **retrieval model** in the `retriever` directory and your **LLM model** in the `models` directory.

You can download the models used in our experiments from [HuggingFace](https://huggingface.co/):

- **Retriever**: [facebook/contriever](https://huggingface.co/facebook/contriever)
- **LLMs**:
  - [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
  - [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
  - [lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)
  - [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)

If you choose to use a different LLM, make sure to update the configuration accordingly in the `model_configs` file.

### 🔹3. Evaluate BEIR

Place your data under the `/datasets` directory. For example, if you're using the NQ dataset, structure it like this: `/datasets/nq`. Make sure both `queries.jsonl` and `qrels/qrels.tsv` are present—these files are required.

Then:

1. Navigate to the `beir_results` directory:

   ```bash
   cd beir_results
   ```

2. Modify the following line in your script to point to your data:

   ```python
   corpus, queries, qrels = GenericDataLoader(data_path, query_file='your_queries.jsonl').load(split='your-qrels')
   ```

   - Replace `query_file` with your actual query file name.
   - Set `split` to match the file in `qrels` (this typically refers to the ground-truth document-query mapping).
      **Note:** Our experiments do not use the ground-truth document directly. You can assign random documents to query IDs, but ensure that query IDs in the qrels file match those in your queries; otherwise, BEIR evaluation will fail.

3. Run the evaluation to generate BEIR results:

   ```bash
   python evaluate_beir.py
   ```

This will create the index mapping between queries and the corpus. It will be used in subsequent experiments to retrieve the top-k documents from the knowledge base.

We have included our experimental BEIR results in the `/beir_results` directory.

### 🔹4. Prepare Arena Data

We generate multiple *plausible but incorrect* answers for each query using the GPT-4o model.

> ⚠️ Before running the following commands, make sure to set your OpenAI API key and configure the correct data path.

To generate incorrect answers, run:

```bash
cd datasets/incorrect_ans
python gen_in_ans.py
```

Once you have the incorrect answers, generate adversarial documents for each of them:

```bash
python gen_adv_doc.py
```

At this point, your arena data is ready! You can refer to our prepared examples in:

- `ms-selected-queries-adv-ans-docs.json`
- `nq-all-adv-docs.json`

**Split Data by Incorrect Answer Position**

Each attack method targets **one incorrect answer at a time**. To support this, we split the dataset based on the position of each incorrect answer.

Run the following command to get $a_{in}^i$ (i.e., each individual incorrect answer variant):

```bash
python process_position_ans.py
```

## 🏟️ PoisonArena

**PoisonArena** is our benchmark designed to evaluate and compare the competitive effectiveness of different attack methods using the **Bradley-Terry model**.

### 📁 Project Structure

Poisoned data and related files are organized as follows:

```bash
├── data/
│   ├── attacked_data/          # Attacked data for single-query attack scenarios
│   ├── serials_data/           # Attacked data for sequential (serials) attack scenarios
│   ├── arena_data/             # Preprocessed data used for general evaluation
│   └── arena_data_poisonarena/ # Data specifically formatted for PoisonArena
├── utils/                      # Utility functions and helpers
├── defence/                    # InstructRAG (ICL) defense prompt examples
├── output/                     # Output directory for evaluation results
├── arena.py                    # Main script to run PoisonArena evaluation
└── build_arena_data.py         # Script to build arena-compatible data

```

### ⚔️ How to Run PoisonArena

Let’s walk through how to use PoisonArena:

#### 1. Prepare Your Data

> **Tips:** Before launching **PoisonArena**, make sure you have completed the attacks for each strategy you plan to include. For example, in our experiments, we first ran attacks using methods such as PoisonedRAG to generate the necessary attack data. This data forms the core of the arena battles.

Ensure that:

- You’ve generated arena-compatible data (see **3. Prepare Arena Data** section).
- **Attack data from various strategies is placed under the correct subdirectories** (e.g., `attacked_data/`, `serials_data/`).
- All required files are properly located.

#### 2. Build the Arena Dataset

Navigate to the project directory and run the dataset construction script. Before running, configure the following parameters inside `build_arena_data.py`:

- `arena_data_path`
- `attacker_data_dir`
- `output_path`

```bash
cd arena
python build_arena_data.py
```

#### 3. Run the Arena Simulation

Once the data is ready, launch the competition by running:

```bash
python arena.py
```

Make sure all necessary arguments are correctly set in the python file or passed via command line.

## 💻 Reproducing Experiments

### 🔹 1. COMBAT

> **Tips Again:** Before launching the **combat**, ensure that you’ve completed the attacks for each strategy you intend to include. This data will be required later in the [**Customizing Attackers**](#👥-customizing-attackers) section.

📁 **Directory**: `combat/`

COMBAT is the core module of our PoisonArena benchmark, designed to simulate multi-attackers poisoning attacks on RAG pipelines.

#### ✅ Setup

```bash
conda create -n combat python=3.10
conda activate combat
cd combat
pip install -r requirements.txt
```

#### 📦 Data Structure

Poisoned data is in `combat/data`, structured like this:

```bash
├── combat_data/
│   ├── order_data/       # For attack order experiments
│   ├── serials_data/     # For sequential (series) attack scenarios
│   └── single_data/      # Main experiments with single questions
├── original/             # Raw data before similarity scoring
└── com_sim_text.py       # Script to compute similarity scores
```

#### 🧩 Using Your Own Data

Prepare your data in this format:

```json
{
  "id": "custom_id",
  "question": "Insert your question here",
  "adv_texts": [{ "context": "Your adversarial document" }],
  "incorrect_answer": "Target poison answer",
  "answer": "Ground truth answer"
}
```

Then modify the path in `com_sim_text.py` and run:

```bash
python combat/data/com_sim_text.py
```

This will generate similarity-scored inputs for combat experiments.

#### ⚔️ Running the Combat

Quick experiment launch:

```bash
cd combat
python run-diff-cor-random.py
```

Results will be saved in:

- `combat_details`: showing the combat details(like question, topk docs, attackers target answers and so on).
- `all_results`: the final combat results. You can see some previous results we obtained there!

The results include metrics such as **ASR** and **F1-score**.

#### 👥 Customizing Attackers

You can set the attackers info in `combat/combat_clean-diff-corpus-random.py` about `all_attackers`.

here is an example(which has also been shown in our code.)

```python
all_attackers = [
    {
        "nick_name": "GASLITE",
        "name": "GASLITE5-ans",
        "base_path": "data/combat_data/single_data/nq/gaslite/gaslite-dcorpus-ans.json",
        "adv_per_query": 5
    }
]
```

Then you can run the code to get your own combat results!

#### 📄 Advanced Experiments

- **Sequential Attacks** (series of related queries):

  ```bash
  python run-diff-cor-random-serials.py
  ```

- **Order-Sensitive Attacks**:

  ```bash
  python run-diff-cor-random-order.py
  ```

- **Defensive RAG with ICL (InstructRAG)**:

  ```bash
  python run-diff-cor-random-serials-defence.py
  ```

🔎 Besides, for the **MS MARCO** dataset, we also include an enhanced evaluation method (**substring match + GPT-3.5 judge!**) to assess combat results — it’s a more accurate and reliable approach in our scenarios! 😊

### 🔹 2. Poison Methods

**Directory**: `poison_methods/`

Each folder contains a minimal, standalone implementation of the respective poisoning method, adapted slightly for our experimental setup.

> ⚠️ For full implementation details and advanced usage, please refer to each method’s original source repository (linked below).
> The modifications we've made are mainly in the data reading and preprocessing; the core attack logic remains unchanged.

#### 2.1 PoisonedRAG

**Directory**: `poisonedrag/`
**Source**: [PoisonedRAG GitHub](https://github.com/sleeepeer/PoisonedRAG)

##### ✅ Setup

```bash
conda create -n PoisonedRAG python=3.10
conda activate PoisonedRAG
cd poisonedrag
pip install -r requirements.txt
```

##### 📄 Data Format

```json
{
    "test58": {
        "id": "test58",
        "question": "Here is the question",
        "correct answer": "this is the ground truth answer",
        "incorrect answer": "the target answer of the attacker",
        "adv_texts": ["here are many adversarial texts"]
    }
}
```

##### 🚀 Run Attack

Set `attack_data_dir` and `attack_data_name`, then execute:

```bash
python PoisonedRAG_gen_adv.py
```

#### 2.2 AdvDec

**Directory**: `advdec/`
**Source**: [Adversarial Decoding GitHub](https://github.com/collinzrj/adversarial_decoding)

##### ✅ Setup

```bash
conda create -n advdec python=3.10
conda activate advdec
cd advdec
pip install -r requirements.txt
```

##### 📄 Data Format

```json
[
    {
        "id": "test58",
        "index_cluster": 0,
        "queries": ["Query 1", "Query 2"],
        "misinfo": ["a lot of misinfo docs!"]
    }
]
```

You can also convert data from PoisonedRAG format using `advdec/datasets/process.py`. (📌 Tip: Don’t forget to rename the files appropriately.)

##### 🚀 Run Attack

Set `attack_data_dir`, `attack_data_name`, and `model_path`, then run:

```bash
python main.py --experiment misinfo
```

##### 🛠️ Process Results

Move the results to `process_data/AdvDec/` and execute:

```bash
python process.py
```

#### 2.3 GARAG

**Directory**: `garag/`
**Source**: [GARAG GitHub](https://github.com/zomss/GARAG)

##### ✅ Setup

```bash
conda create -n garag python=3.10
conda activate garag
cd garag
pip install -r requirements.txt
```

##### 📄 Data Format

```json
[
    {
        "id": "test58",
        "question": "Here is the question",
        "answers": ["Answer1", "Answer2"],
        "ctxs": [
            {
                "id": "doc1242987",
                "has_answer": true,
                "score": 1.44,
                "context": "Context"
            }
        ]
    }
]
```

> **Note**: `ctxs` are the top retrieved documents from the knowledge base.

##### 🚀 Run Attack

Execute:

```bash
sh eval.sh
```

If you encounter model path errors, check `GARAG/src/option.py`.

##### 🛠️ Process Results

Use the script `GARAG/process_G.py` to reformat output.

#### 2.4 GASLITE

**Directory**: `gaslite/`
**Source**: [GASLITE GitHub](https://github.com/matanbt/gaslite)

##### ✅ Setup

```bash
conda create -n garag python=3.10
conda activate garag
cd gaslite
pip install -r requirements.txt
```

##### 📄 Data Format

Same format as PoisonedRAG.

##### 🚀 Run Attack

For a **single query** attack:

```bash
sh ./scripts/attack0_knows-all.sh
```

For **serial query** attacks:

```bash
sh ./scripts/attack2_knows-nothing.sh 
```

> We've modified the original script to accommodate our own data reading and formatting.

##### 🛠️ Process Results

Processed via: `combat/process_data/gaslite/`

#### 2.5 Corpus-Poison

**Directory**: `corpus_poison/`
**Source**: [Corpus Poisoning GitHub](https://github.com/princeton-nlp/corpus-poisoning)

##### ✅ Setup

You can use the same environment as PoisonedRAG.

##### 📄 Data Format

Same format as PoisonedRAG.

##### 🚀 Run Attack

Both commands below serve the same purpose in our setup:

```bash
sh scripts/attack_poison.sh contriever nq 1
```

> The “1” indicates that each cluster corresponds to a specific query or a group of paraphrased queries, as our experiments are query-focused (unlike the full-dataset approach of the original repo).

##### 🛠️ Process Results

Use the python file located at: `combat/process_data/corpus-poison/`

#### 2.6 Content-Poison

**Directory**: `content_poison/`
**Source**: [Content Poisoning GitHub](https://github.com/ZQ-Struggle/Content-Poisoning)

##### ✅ Setup

```bash
conda create -n cpoison python=3.10
conda activate cpoison
cd content_poison
pip install -r requirements.txt
```

##### 📄 Data Format

```
prompt^^target^^fail_flag^^succ_flag^^control^^id
<Instruction>...</Instruction>/n<Known information>{CONTEXT1}^@^&^&{CONTEXT2}&^&... </Known information>/n<Question>...</Question>/n<Answer>^^...
```

> `^@^` is where the trigger is inserted (here is after the top 1 document).
> `&^&` separates individual documents.

##### 🚀 Run Attack

For a **single query** attack:

```bash
sh ./scripts/sequence_generation_word_level.sh llama3_8b nq-inco_ans_1
```

> Here, `nq-inco_ans_1` refers to the input file `nq-inco_ans_1.tsv`.

##### 🛠️ Process Results

Use the script at: `content_poison/extract/format_output.py`

## 🙏 Acknowledgements

Thanks again to all the incredible works we built upon:

- [PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG)
- [AdvDec](https://github.com/collinzrj/adversarial_decoding)
- [GARAG](https://github.com/zomss/GARAG)
- [GASLITE](https://github.com/matanbt/gaslite)
- [Corpus-Poison](https://github.com/princeton-nlp/corpus-poisoning)
- [Content-Poisoning](https://github.com/ZQ-Struggle/Content-Poisoning)
- [InstructRAG](https://github.com/weizhepei/InstructRAG)

We are also grateful to both open-source contributors and proprietary model providers, including OpenAI's ChatGPT, whose models supported our experiments.
