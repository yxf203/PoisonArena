import torch
import requests
import time
def query_gpt(input, model_name, return_json: bool):

    url="https://api.openai.com/v1/chat/completions"
    API_KEY="YOUR_API_KEY" # Replace with your actual API key

    headers = {
        'Authorization': f"Bearer {API_KEY}",
        'Content-Type': 'application/json'
    }
    data = {
        'model': model_name,
        'temperature': 0,
        'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'}, 
                     {'role': 'user', 'content': input}]
    }
    if return_json:
        data['response_format'] = {"type": "json_object"}
    
    max_retries=10
    retry_delay=5
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result_json = response.json()
            print("is using model:", result_json.get('model'))
            return result_json['choices'][0]['message']['content']
        
        except requests.exceptions.ReadTimeout as e:
            print(f"Attempt {attempt}: Read timeout. Retrying in {retry_delay} seconds...")
            if attempt == max_retries:
                raise e
            time.sleep(retry_delay)

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            raise e


def compute_sim(model, tokenizer, get_emb, query, doc, score_function, device):
    # Ensure both query and text are moved to the same device as the model (GPU in this case)
    # query_input = query.to(device)
    # text_input = text.to(device)
    query = tokenizer(query, padding=True, truncation=True, return_tensors="pt")
    
    # Move the query tensor to the same device as the model
    query = {key: value.to(device) for key, value in query.items()}
    doc = tokenizer(doc, padding=True, truncation=True, return_tensors="pt")
    
    # Move the document tensor to the same device as the model
    doc = {key: value.to(device) for key, value in doc.items()}

    query_emb = get_emb(model, query)
    adv_emb = get_emb(model, doc)
    
    if score_function == 'dot':
        # Dot product similarity
        adv_sim = torch.mm(adv_emb, query_emb.T).cpu().item()
    elif score_function == 'cos_sim':
        # Cosine similarity
        adv_sim = torch.cosine_similarity(adv_emb, query_emb).cpu().item()
    return adv_sim