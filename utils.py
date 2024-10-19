#coding=utf-8
import json
import os
import torch
import numpy as np
import random
import re
from eval_script import get_entities


def read_dataset(path):
    if 'jsonl' in path:
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                dataset.append(json.loads(line))
    elif 'json' in path:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        if isinstance(dataset, dict):
            if 'data' in dataset:
                dataset = dataset['data']
    else:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = f.readlines()
    return dataset


def save_dataset(path, dataset):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

def replace_punctuation(str):
    return str.replace("\"", "").replace("'", "")
import os

# Temporary fix for bug where {}^<\` characters roundtrip into \u2047 (??) character
def fix_buggy_characters(str):
    return re.sub("[{}^\\\\`\u2047<]", " ", str)

def score_string_similarity(str1, str2):
    if str1 == str2:
        return 3.0   # Better than perfect token match
    str1 = fix_buggy_characters(replace_punctuation(str1))
    str2 = fix_buggy_characters(replace_punctuation(str2))
    if str1 == str2:
        return 2.0
    if " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        if str1 == str2:
            return 1.0
        else:
            return 0.0



def read_msqa(path):
    dataset = read_dataset(path)
    dataset_new = []
    for sample in dataset:
        id = sample['id']
        question = sample['question']
        context = sample['context']
        label = sample['label']
        answers_w_idx = get_entities(label, context)
        answers_w_idx = sorted(answers_w_idx, key=lambda x: x[1])
        answers = [item[0] for item in answers_w_idx]
        context_char = ""
        context_char_idx_beg, context_char_idx_end = [], []
        beg_idx = 0
        for word in context:
            context_char_idx_beg.append(beg_idx)
            context_char_idx_end.append(beg_idx + len(word))
            beg_idx += len(word) + 1
            context_char += word + ' '
        context_char = context_char.strip()

        answers_idx_char = []
        for ans, beg_idx, end_idx in answers_w_idx:
            # if context_char[context_char_idx_beg[beg_idx]: context_char_idx_end[end_idx]] != ans:
            #     print(context_char[context_char_idx_beg[beg_idx]: context_char_idx_end[end_idx]])
            #     print(ans)
            assert context_char[context_char_idx_beg[beg_idx]: context_char_idx_end[end_idx]] == ans
            answers_idx_char.append([
                context_char_idx_beg[beg_idx],
                context_char_idx_end[end_idx],
            ])

        dataset_new.append({
            'id': id,
            'question': ' '.join(question),
            'context': context_char,
            'answers': answers,
            'sample': sample,
            'answers_idx': answers_idx_char
        })
    return dataset_new

def save_model(output_model_file, model, optimizer=None):
    os.makedirs(output_model_file, exist_ok=True)
    output_model_file += 'pytorch_model.bin'
    torch.save({
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict()
    }, output_model_file, _use_new_zipfile_serialization=False)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # gpu
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
