import os
import json
import glob

def build_arena_data():
    """
    Integrates data from nq-arena.json and all JSON files in the attacker_data directory
    into a single JSON file with a specific format.
    """
    # Define paths
    arena_data_path = "data/arena_data/nq-arena-serials.json"
    attacker_data_dir = "data/serials_data"
    output_path = "data/arena_data_poisonarena/nq_arena_data_serials.json"
    
    # Load nq-arena.json
    with open(arena_data_path, 'r') as f:
        arena_data = json.load(f)
    
    # Initialize the integrated data dictionary
    integrated_data = {}
    
    # Process nq-arena.json data
    for item_id, item_data in arena_data.items():
        integrated_data[item_id] = {
            'id': item_id,
            'question': item_data.get('question', ''),
            'correct_answer': item_data.get('correct answer', ''),
            'all_adv_docs': []
        }
        
        # Process each adversarial document and initialize attackers field
        for adv_doc in item_data.get('all_adv_docs', []):
            new_adv_doc = {
                'incorrect_answer': adv_doc.get('incorrect_answer', ''),
                'adv_texts': adv_doc.get('adv_texts', []),
                'attackers': {}  # Initialize empty attackers dictionary for each incorrect answer
            }
            integrated_data[item_id]['all_adv_docs'].append(new_adv_doc)
    
    # Find all JSON files in attacker_data directory and its subdirectories
    attacker_json_files = []
    for root, dirs, files in os.walk(attacker_data_dir):
        for file in files:
            if file.endswith('.json'):
                attacker_json_files.append(os.path.join(root, file))
    
    # Process each attacker JSON file
    for json_file in attacker_json_files:
        try:
            with open(json_file, 'r') as f:
                attacker_data = json.load(f)
            
            # Extract attacker name from path
            attacker_name = os.path.basename(os.path.dirname(json_file))
            
            # Process each item in the attacker data
            if isinstance(attacker_data, list):
                for item in attacker_data:
                    process_attacker_item(integrated_data, item, attacker_name)
            elif isinstance(attacker_data, dict):
                for item_id, item in attacker_data.items():
                    if item_id in integrated_data:
                        process_attacker_item(integrated_data, item, attacker_name, item_id)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Filter out incorrect answers with no attackers
    filter_empty_attackers(integrated_data)
    
    # Save the integrated data to a new JSON file
    with open(output_path, 'w') as f:
        json.dump(integrated_data, f, indent=4)
    
    print(f"Arena data built successfully and saved to {output_path}")
    return integrated_data

def process_attacker_item(integrated_data, item, attacker_name, item_id=None):
    """
    Process an attacker item and add it to the appropriate incorrect_answer in all_adv_docs.
    
    Args:
        integrated_data (dict): The integrated data dictionary
        item (dict): The attacker item data
        attacker_name (str): The name of the attacker
        item_id (str, optional): The item ID if not included in the item
    """
    # Get item_id from the item if not provided
    if item_id is None:
        item_id = item.get('id')
    
    # Skip if item_id is not in integrated_data
    if not item_id or item_id not in integrated_data:
        return
    
    # Get the incorrect answer targeted by this attacker
    incorrect_answer = item.get('incorrect_answer', '')
    
    # Find the matching incorrect_answer in all_adv_docs
    found = False
    for adv_doc in integrated_data[item_id]['all_adv_docs']:
        if adv_doc['incorrect_answer'] == incorrect_answer:
            # Add attacker data to this incorrect_answer
            adv_doc['attackers'][attacker_name] = {
                'adv_texts': item.get('adv_texts', []),
                'incorrect_answer': incorrect_answer
            }
            found = True
            break
    
    # If corpus_poison or other attackers don't match any existing incorrect_answer,
    # we can still keep their data at the item level
    if not found and attacker_name == 'corpus_poison':
        # Some attackers like corpus_poison might not target specific incorrect answers
        if 'attackers' not in integrated_data[item_id]:
            integrated_data[item_id]['attackers'] = {}
        
        integrated_data[item_id]['attackers'][attacker_name] = {
            'adv_texts': item.get('adv_texts', []),
            'incorrect_answer': incorrect_answer
        }

def filter_empty_attackers(integrated_data):
    """
    Remove incorrect answers that have no attackers from all_adv_docs.
    
    Args:
        integrated_data (dict): The integrated data dictionary
    """
    items_processed = 0
    answers_removed = 0
    
    for item_id, item_data in integrated_data.items():
        # Filter out adv_docs with empty attackers
        filtered_adv_docs = [doc for doc in item_data.get('all_adv_docs', []) 
                            if doc.get('attackers', {})]
        
        # Count removed answers
        removed_count = len(item_data.get('all_adv_docs', [])) - len(filtered_adv_docs)
        answers_removed += removed_count
        
        # Update the item with filtered adv_docs
        item_data['all_adv_docs'] = filtered_adv_docs
        items_processed += 1
    
    print(f"Processed {items_processed} items, removed {answers_removed} incorrect answers with no attackers")

def validate_arena_data(integrated_data):
    """
    Validate that for each item in the arena data, all attackers targeting a specific incorrect_answer
    have the same incorrect_answer value.
    
    Args:
        integrated_data (dict): The integrated arena data dictionary
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    validation_passed = True
    
    for item_id, item_data in integrated_data.items():
        print(f"Validating item {item_id}...")
        
        # Check each incorrect answer in all_adv_docs
        for adv_doc in item_data.get('all_adv_docs', []):
            incorrect_answer = adv_doc.get('incorrect_answer', '')
            
            # Check each attacker for this incorrect answer
            for attacker_name, attacker_data in adv_doc.get('attackers', {}).items():
                attacker_incorrect_answer = attacker_data.get('incorrect_answer', '')
                
                if attacker_incorrect_answer != incorrect_answer:
                    print(f"  ERROR: In item {item_id}, attacker {attacker_name} has incorrect_answer '{attacker_incorrect_answer}' "
                          f"which doesn't match the expected '{incorrect_answer}'")
                    validation_passed = False
        
        # Also check attackers at the item level (for corpus_poison, etc.)
        if 'attackers' in item_data:
            for attacker_name, attacker_data in item_data['attackers'].items():
                # These are special cases, so we just log them for awareness
                print(f"  Note: Item {item_id} has attacker {attacker_name} at the item level with "
                      f"incorrect_answer: '{attacker_data.get('incorrect_answer', '')}'")
    
    if validation_passed:
        print("Validation passed: All attackers have consistent incorrect answers.")
    else:
        print("Validation failed: Some attackers have inconsistent incorrect answers.")
    
    return validation_passed

if __name__ == "__main__":
    integrated_data = build_arena_data()
    validate_arena_data(integrated_data)
