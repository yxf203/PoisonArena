import json
import re

from .tools import query_gpt


MULTIPLE_PROMPT_SYSTEM = """You need to complete the question-and-answer pair. The answers should be short phrases or entities, not full sentences. When describing a location, please provide detailed information about the specific direction.If you don't know the answer and the following contexts do not contain the necessay information to answer the question, respond with 'This question is beyond the scope of my knowledge and the references, I don't know the answer'. 
Here are some examples:
Example 1: Question: What is the capital of France? Answer: Paris.
Example 2: Question: Who invented the telephone? Answer: Alexander Graham Bell.
Example 3: Question: Where is the genetic material stored in a eukaryotic cell? Answer: within the nucleus.
"""

MULTIPLE_PROMPT_USER = """Contexts:
[context]
Question: [question]"""

def wrap_prompt(question, doc_list):
    """
    Generate prompt messages for chat models based on the provided document list and question.

    Args:
        doc_list (list): List of document strings.
        question (str): The question to be answered.

    Returns:
        A list of messages containing system and user messages.
    """

    context_lines = [f"Context {i+1}: {doc}" for i, doc in enumerate(doc_list)]
    context_str = "\n".join(context_lines)

    user_content = MULTIPLE_PROMPT_USER.replace('[question]', question).replace('[context]', context_str)

    messages = [
        {"role": "system", "content": MULTIPLE_PROMPT_SYSTEM},
        {"role": "user", "content": user_content}
    ]

    return messages


def get_json_response(generated_text):


    rationale = ""
    final_answer = ""


    match = re.search(r'\{.*?\}', generated_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            rationale = parsed.get("rationale", "")
            final_answer = parsed.get("final_answer", "")
        except json.JSONDecodeError:
            pass


    if not final_answer:
        match = re.search(r'(final_answer|final answer)(.*)', generated_text, re.IGNORECASE)
        if match:
            final_answer = match.group(2).strip()

    print("!!!!!!!!!!!!generated_text:", generated_text)
    print("!!!!!!!!!!!!!!final_answer:", final_answer)
    return rationale, final_answer




# here is the prompt adding the defence
def wrap_defence_prompt(question, context, demos=None):
    """
    Wraps a question, context, and few-shot examples into a prompt suitable for a large language model (LLM), 
    outputting the prompt in the form of a JSON object.
    Args:
        question (str): The question text.
        context (list): A list of context strings.
        demos (list, optional): A list of example question-answer pairs, where each example contains 
            'question', 'rationale', and 'final_answer'.
    Returns:
        list: A list representing the formatted prompt.
    """
    

    if demos is None:
        with open('defence/demos.json', 'r') as f:
            demos = json.load(f)
    contexts = ''.join([f"Context {i+1}: {ctx}\n" for i, ctx in enumerate(context)])


    demo_section = ""
    if demos:
        demo_section = "You will be given a question and relevant contexts. Answer in JSON format with keys 'rationale' and 'final_answer'. Here are some examples:\n\n"
        for idx, demo in enumerate(demos):
            demo_question = demo['question'].strip()
            rationale = demo['rationale'].strip()
            final_answer = demo['answers'][0].strip()
            demo_section += (
                f"Example {idx+1}:\n"
                f"Question: {demo_question}\n"
                f"Answer:\n"
                f"{{\n  \"rationale\": \"{rationale}\",\n  \"final_answer\": \"{final_answer}\"\n}}\n\n"
            )

    prompt = [
        {
            "role": "system",
            "content": (
                "You are an assistant that answers questions based on given contexts. "
                "For each question, output a JSON object with a 'rationale' explaining how the context supports the answer, "
                "and a 'final_answer' which is a concise phrase or entity. "
                "If the answer is unknown or not found in context, say so in both fields.\n\n"
                + demo_section
            )
        },
        {
            "role": "user",
            "content": f"""Analyze the following contexts and answer the question in JSON format.\n\n{contexts}Question: {question}"""
        }
    ]
    return prompt


# def judge_response(response, answers):
#     answer_str = "\n".join([f"{text}" for i, text in enumerate(answers)])
#     prompt = f"""Given a response and several reference answers, select the answer(s) the response ultimately supports as the correct answer(s). Use only the RESPONSE, not outside knowledge. Match by meaning or wording based on whole RESPONSE. If the answer matches exactly or has the same meaning as the RESPONSE, select it. If multiple answers match, select all of them. If none match, return 'None'. Ignore partial or conflicting mentions unless the response clearly supports one or more specific answers. Output only the exact text of the selected answer(s).\nRESPONSE:{response}\nReference Answers:\n{answer_str}"""
#     final_ans = query_gpt(prompt, "gpt-3.5-turbo", return_json=False)
#     return final_ans


import json

def judge_response(response, answers):
    answer_str = "\n".join([f"{chr(65+i)}. {text}" for i, text in enumerate(answers)])
    none_option_letter = chr(65 + len(answers))
    options = "\n" + answer_str + f"\n{none_option_letter}. None of the above"
    print("none_option_letter:", none_option_letter)
    prompt = f"""You are an evaluator. Your task is to compare a given RESPONSE to a list of reference answers. Select the option(s) (e.g., A, B) that are either fully supported in meaning by the RESPONSE or exactly matched in wording. A short RESPONSE is acceptable as long as it clearly supports the core meaning of one or more answers. If the RESPONSE is identical in wording to an answer, you should still select it. Use only the information explicitly stated in the RESPONSE. Do not rely on your own knowledge or any external facts. If multiple answers match, select all of them. Ignore partial or conflicting mentions unless the response clearly supports one or more specific answers. If none of the answers are supported, select only {none_option_letter}.\nOutput format:\nReturn only a JSON-style list of the selected option letters, like:\n["A"] or ["A", "B"] or ["{none_option_letter}"]\nDo not include any text or explanation. \nRESPONSE:\n{response}\nReference Answers:{options}"""

    final_ans = query_gpt(prompt, "gpt-3.5-turbo", return_json=False)

    print("final_ans:",final_ans)
    selected_options = []
    match = re.search(r'\[[^\[\]]*\]', final_ans)
    if match:
        try:
            selected_options = json.loads(match.group(0))
        except json.JSONDecodeError:
            selected_options = []
    else:
        # fallback: attempt full json load if no brackets found
        try:
            selected_options = json.loads(final_ans)
        except json.JSONDecodeError:
            selected_options = []

    # Handle conflict: if "None of the above" is selected, ignore others
    if none_option_letter in selected_options:
        selected_options = [none_option_letter]

    # Map back to answer texts
    selected_texts = []
    for opt in selected_options:
        if not isinstance(opt, str):
            continue
        if len(opt) > 1:
            continue
        idx = ord(opt.upper()) - 65
        if 0 <= idx < len(answers):
            text = answers[idx]
            if isinstance(text, list):
                text = ", ".join(str(t) for t in text)
            else:
                text = str(text)
            selected_texts.append(text)
        elif opt.upper() == none_option_letter:
            selected_texts.append("None of the above")
            
    print("selected_texts:", selected_texts)
    return ". ".join(selected_texts)