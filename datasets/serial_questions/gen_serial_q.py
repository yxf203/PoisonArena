import argparse
import os
import json

import requests
import pdb
from tqdm import tqdm


def query_gpt(input, model_name, return_json: bool):
    url = "https://api.openai.com/v1/chat/completions"
    API_KEY = "Your API Key"
    headers = {
        'Authorization': f"Bearer {API_KEY}",
        'Content-Type': 'application/json'
    }
    data = {
        'model': model_name,
        'temperature': 1,
        'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'}, 
                     {'role': 'user', 'content': input}]
    }
    if return_json:
        data['response_format'] = {"type": "json_object"}
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    # pdb.set_trace()
    result = {'usage': response.json()['usage'], 'output': response.json()['choices'][0]['message']['content']}
    return result['output']


def parse_args():
    parser = argparse.ArgumentParser(description="test")

    # Retriever and BEIR datasets
    parser.add_argument(
        "--eval_model_code",
        type=str,
        default="contriever",
        choices=["contriever-msmarco", "contriever", "ance"],
    )

    # data path
    parser.add_argument("--data_path", type=str, default="datasets/serial_questions/nq-all-adv-docs.json")
    parser.add_argument("--save_path", type=str, default="datasets/serial_questions/")

    # model 
    parser.add_argument("--model_name", type=str, default="gpt-4o")

    # parameters
    parser.add_argument("--num_serial_q", type=int, default=10, help="number of serial questions")

    args = parser.parse_args()
    return args


def gen_serial_q(args):
    '''Use qrels (ground truth contexts) to generate a correct answer for each query and then generate an incorrect answer for each query'''

    with open(args.data_path, 'r') as f:
        all_data = json.load(f)

    for query_id, query_data in tqdm(all_data.items()):
        question = query_data["question"]
        correct_answer = query_data["correct answer"]

        print(query_id)
        print(question)
        print(correct_answer)
        serial_q_list = []

        gen_serial_q_prompt = f"""
        Given a question and its corresponding answer, your task is to rewrite the question to create new versions without changing the answer. Without changing the answer, create as many varied forms of the question as possible.

        ### Task:
        - Given a question and its corresponding answer, generate {args.num_serial_q} different questions without changing the answer.
        - Without changing the answer, create as many varied forms of the question as possible.

        ### Example:
        - Question: who is the girl ray in star wars?
        - Answer: Daisy Ridley
        - Serial Questions: 
            - Which actress plays the character Rey in Star Wars?
            - Who portrays Rey in the Star Wars series?
            - The role of Rey in Star Wars was played by whom?
            - ...
            - Who was cast as Rey in the Star Wars movies?

        ### Input:
        - Question: {question}.  
        - Answer: {correct_answer}.  

        ### Output Format:
        Provide your response in valid JSON format with the following structure:
        {{
            "serial_questions": [
                "serial_question1",
                "serial_question2",
                ...
                "serial_question{args.num_serial_q}"
            ]
        }}
        """


        response = query_gpt(gen_serial_q_prompt, model_name=args.model_name, return_json=True)

        if isinstance(response, str):
            serial_q_list = json.loads(response) 
        else:
            serial_q_list = response


        serial_q_list = serial_q_list.get("serial_questions", []) 
        if len(serial_q_list) < args.num_serial_q:
            print(f"{query_id} number of serial questions less than {args.num_serial_q}!")
        all_data[query_id]["serial_questions"] = serial_q_list


    json.dump(all_data, open(os.path.join(args.save_path, f'nq-all-adv-docs-with-serial-q.json'), 'w'), indent=4)



if __name__ == "__main__":
    args = parse_args()
    gen_serial_q(args)
