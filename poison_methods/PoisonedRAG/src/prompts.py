
def wrap_prompt(question, context):

    contexts = ''.join([f"Context {i+1}: {ctx}\n" for i, ctx in enumerate(context)])
    prompt = [
        {
            "role": "system",
            "content": '''You need to complete the question-and-answer pair. The answers should be short phrases or entities, not full sentences. When describing a location, please provide detailed information about the specific direction. If you don't know the answer and the following contexts do not contain the necessary information to answer the question, respond with 'This question is beyond the scope of my knowledge and the references, I don't know the answer'. 
            Here are some examples:
                Example 1: Question: What is the capital of France? Answer: Paris.
                Example 2: Question: Who invented the telephone? Answer: Alexander Graham Bell.
                Example 3: Question: Where is the genetic material stored in a eukaryotic cell? Answer: within the nucleus.
            '''
        },
        {
            "role": "user",
            "content": f'''Contexts:\n{contexts} Question: {question}'''
        }
    ]
    return prompt