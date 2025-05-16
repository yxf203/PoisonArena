def find_token(tokenizer, sentence_ids, sentence):        
    
    # subsentence = subsentence.replace(' ', '')
    # subsentence = sentence.replace(' ', '')
    # subsentence = sentence
    # print("sentence_ids: ", sentence_ids)
    # print("subsentence: ", subsentence)
    results = []
    current = 0
    # print("tokenizer.name_or_path: ", tokenizer.name_or_path)
    # subsentence = tokenizer.decode(tokenizer(sentence).input_ids).replace("<s> ", "").strip()
    if "llama" in tokenizer.name_or_path.lower():
        subsentence = tokenizer.decode(tokenizer(sentence).input_ids).replace("<|begin_of_text|>", "").strip()
    else:
        subsentence = tokenizer.decode(tokenizer(sentence).input_ids).replace("<s> ", "").strip()

    while True:
        diff_start = 0
        diff_end = 0
        flag = False
        for idx, tok in enumerate(sentence_ids[current:]): # 259:' ' , 29871: ' '
            subword = tokenizer.decode(tok)
            # print("subsentence: ", subsentence)
            # print("subword: ", subword)
            if subword == '':
                subword = " "
            if subsentence.startswith(subword) or (subword != " " and subsentence.startswith(subword.strip())):
                # print("here here!")
                # print("subsentence:", subsentence)
                # print(tokenizer.decode(sentence_ids[idx+current:]))
                if tokenizer.decode(sentence_ids[idx+current:]).strip().startswith(subsentence):
                    flag = True
                    diff_start = idx+current
                    break
        # print(flag, diff_start)
        if flag is False or diff_start == 0:
            break
        
        for idx, tok in enumerate(sentence_ids[diff_start:]):
            cur_sentence = tokenizer.decode(sentence_ids[diff_start:diff_start+idx+1])
            cur_sentence_shorter = tokenizer.decode(sentence_ids[diff_start:diff_start+idx])
            # print("cur_sentence: ", cur_sentence)
            # print("cur_sentence_shorter: ", cur_sentence_shorter)
            if len(cur_sentence) >= len(subsentence) and len(cur_sentence_shorter) < len(subsentence):
                diff_end = diff_start + idx + 1
                current = diff_end + 1
                break
        
        results.append((diff_start, diff_end))
        # print('results:',results)
        assert flag is True and diff_end != 0, "why flag is True but the end is not found?"
        # if flag is False or diff_end == 0:
            # return -1, -1

    return results
    

# def find_token(tokenizer, sentence_ids, subsentence):        
#     diff_start = 0
#     diff_end = 0
#     flag = False
#     # subsentence = subsentence.replace(' ', '')
#     for idx, tok in enumerate(sentence_ids): # 259:' ' , 29871: ' '
#         subword = tokenizer.decode(tok)
#         if subword == '':
#             subword = " "
#         if subsentence.startswith(subword):
#             if tokenizer.decode(sentence_ids[idx:]).startswith(subsentence):
#                 flag = True
#                 diff_start = idx
#                 break
#     if flag is False or diff_start == 0:
#         return -1, -1
#     for idx, tok in enumerate(sentence_ids[diff_start:]):
#         cur_sentence = tokenizer.decode(sentence_ids[diff_start:diff_start+idx+1])
#         cur_sentence_shorter = tokenizer.decode(sentence_ids[diff_start:diff_start+idx])
#         if len(cur_sentence) >= len(subsentence) and len(cur_sentence_shorter) < len(subsentence):
#             diff_end = diff_start + idx
#             break
#     if flag is False or diff_end == 0:
#         return -1, -1
#     else:
#         return diff_start, diff_end


# def find_token(tokenizer, sentence_ids, subsentence):        
#     diff_start = 0
#     diff_end = 0
#     flag = 0
#     # subsentence = subsentence.replace(' ', '')
#     for idx, tok in enumerate(sentence_ids): # 259:' ' , 29871: ' '
#         subword = tokenizer.decode(tok)
#         if subword == '':
#             subword = " "
#         if flag==0 and subsentence.startswith(subword):
#             flag += len(subword)
#             diff_start = idx
#         elif flag > 0 and subsentence[flag:].startswith(subword):
#             flag += len(subword)
#         else:
#             flag = 0
#         if flag == len(subsentence):  #todo: example https://aaa.bb.com/  <https://aaa.bb.com/>.  /> will be tokenized as one token, making the comparaing fail. 
#             diff_end = idx
#             break
#     if flag == 0 or diff_end == -1:
#         return -1, -1
#     else:
#         return diff_start, diff_end