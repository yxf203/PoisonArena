import json

# 文件路径
file_ans1 = 'cpoison5-serials-ans1.json'
file_ans2 = 'cpoison5-serials-ans2.json'

# 读取 JSON 文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
# 将列表转换为以 id 为键的字典
def list_to_dict(data_list):
    return {item['id']: item for item in data_list}

# 找出 ans1 比 ans2 多的 ID
def find_extra_ids(ans1, ans2):
    ids_ans1 = set(ans1.keys())
    ids_ans2 = set(ans2.keys())
    extra_ids = ids_ans1 - ids_ans2
    return extra_ids

def main():
    # 加载数据
    ans1 = load_json(file_ans1)
    ans2 = load_json(file_ans2)
    ans1 = list_to_dict(ans1)
    ans2 = list_to_dict(ans2)
    # 找出 ans1 中重复的 ID
    seen_ids = set()
    duplicate_ids = set()
    for item in ans1.values():
        if item['id'] in seen_ids:
            duplicate_ids.add(item['id'])
        else:
            seen_ids.add(item['id'])
    
    # 输出重复的 ID
    print("Duplicate IDs in ans1:")
    for _id in duplicate_ids:
        print(_id)
    # 找出多余的 ID
    extra_ids = find_extra_ids(ans1, ans2)
    
    # 输出结果
    print("IDs in ans1 but not in ans2:")
    for _id in extra_ids:
        print(_id)

if __name__ == '__main__':
    main()