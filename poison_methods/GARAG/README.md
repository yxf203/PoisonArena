# GARAG
Official Repository for the paper ["Typos that Broke the RAG's Back: Genetic Attack on RAG Pipeline by Simulating Documents in the Wild via Low-level Perturbations"](https://arxiv.org/abs/2404.13948) (Findings of EMNLP 2024)

## Abstract

<div align="center">
  <img alt="Motivation of GARAG" src="./images/motivation.png" width="400px">
</div>

The robustness of recent Large Language Models (LLMs) has become increasingly crucial as their applicability expands across various domains and real-world applications.Retrieval-Augmented Generation (RAG) is a promising solution for addressing the limitations of LLMs, yet existing studies on the robustness of RAG often overlook the interconnected relationships between RAG components or the potential threats prevalent in real-world databases, such as minor textual errors.In this work, we investigate two underexplored aspects when assessing the robustness of RAG: 1) vulnerability to noisy documents through low-level perturbations and 2) a holistic evaluation of RAG robustness. Furthermore, we introduce a novel attack method, the Genetic Attack on RAG (*GARAG*), which targets these aspects.Specifically, *GARAG* is designed to reveal vulnerabilities within each component and test the overall system functionality against noisy documents. We validate RAG robustness by applying our *GARAG* to standard QA datasets, incorporating diverse retrievers and LLMs. The experimental results show that *GARAG* consistently achieves high attack success rates. Also, it significantly devastates the performance of each component and their synergy, highlighting the substantial risk that minor textual inaccuracies pose in disrupting RAG systems in the real world.

## Installation
The first step is to create a conda environment as follows:
```bash
$ conda env create --file environment.yaml
```

## Supported Model

We support DPR and Contriever as retrievers, and Llama-2, Vicuna, and Mistral as readers within our RAG system. You can access the model via the following link. Our code currently requires that the model be downloaded to a specific directory, which will be used as a parameter when running the main code.

| Retriever | Question Encoder | Document Encoder |
| --- | --- | --- |
| DPR | [dpr-question_encoder-multiset-base](https://huggingface.co/facebook/dpr-question_encoder-multiset-base) | [dpr-ctx_encoder-multiset-base](https://huggingface.co/facebook/dpr-ctx_encoder-multiset-base) |
| Contriever | [contriever](https://huggingface.co/facebook/contriever) | same with question encoder | 

| Reader | Link |
| --- | --- |
| Llama-2-7b |[Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| Llama-2-13b | [Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) | 
| vicuna-7b | [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) |
| vicuna-13b | [vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5) |
| Mistral-7b | [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) | 
