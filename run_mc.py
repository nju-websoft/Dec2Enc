# coding=utf-8
from random import choice

from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizerFast, AlbertTokenizerFast, DebertaTokenizerFast, AutoModel, get_linear_schedule_with_warmup, AutoConfig
from tqdm import trange
import os
import random
import torch
from utils import save_dataset, set_seed, save_model, read_dataset
import json
import argparse
from torch import nn
import math
from collections import OrderedDict
from eval_script import multi_span_evaluate
import copy
import ast
from eval_script import get_entities
import numpy as np
from deepspeed_config import get_train_ds_config
import deepspeed
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

device = torch.device("cuda:0")

class MyDataset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # print('sample:',sample)
        return sample

class MLP(nn.Module):
    def __init__(self, dim0, dim1):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(dim0, dim0)
        self.linear2 = nn.Linear(dim0, dim1)
        self.activate = nn.ReLU()

    def forward(self, input):
        input = self.linear1(input)
        input = self.activate(input)
        input = self.linear2(input)
        return input

class Dec2Enc(nn.Module):
    def __init__(self, model_path, vanilla):
        super(Dec2Enc, self).__init__()

        self.model_path = model_path
        config = AutoConfig.from_pretrained(model_path)
        config._attn_implementation = 'eager'
        if 'bert' in model_path.lower() or 'xlm' in model_path.lower():
            self.model = AutoModel.from_pretrained(model_path, config=config)
        else:
            if vanilla:
                self.model = AutoModel.from_pretrained(model_path, config=config)
            else:
                if 'qwen' in model_path.lower():
                    from models.modeling_qwen2 import Qwen2ForCausalLM
                    self.model = Qwen2ForCausalLM.from_pretrained(model_path, config=config)
                elif 'gemma' in model_path.lower():
                    from models.modeling_gemma2 import Gemma2ForCausalLM
                    self.model = Gemma2ForCausalLM.from_pretrained(model_path, config=config)
                elif 'mistral' in model_path.lower():
                    from models.modeling_mistral import MistralForCausalLM
                    self.model = MistralForCausalLM.from_pretrained(model_path, config=config)
                else:
                    from models.modeling_llama import LlamaForCausalLM
                    self.model = LlamaForCausalLM.from_pretrained(model_path, config=config)
        self. linear = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids,  attention_mask, labels=None):
        if 'bert' in self.model_path.lower() or 'xlm' in self.model_path.lower():
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 # attn_implementation="eager",
                                 output_hidden_states=True,
                                 # use_cache=False,
                                 output_attentions=True,
                                 return_dict=True)
        else:
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 # attn_implementation="eager",
                                 output_hidden_states=True,
                                 use_cache=False,
                                 output_attentions=True,
                                 return_dict=True)
        sequence_output = outputs.hidden_states[-1]
        sequence_output = torch.mean(sequence_output, dim=1)
        logits = self.linear(sequence_output)
        logits = logits.view(-1, 4)
        if labels is not None:
            loss_fun = nn.CrossEntropyLoss()
            loss = loss_fun(logits, labels)
            return loss
        else:
            _, pred = torch.max(logits, dim=-1)
            pred = pred.cpu().tolist()
            return pred



def get_input_feature(features, max_source_length):

    input_texts, labels = [], []
    for sample in features:
        context = sample['context']
        question = sample['question']
        choice = sample['choice']
        answer = sample['answer']
        for item in choice:
            input_text = f'Question: {question} \n Option: {item} \n Context: {context}'
            input_texts.append(input_text)
        labels.append(answer)

    encoding = tokenizer(input_texts,
                         padding='longest',
                         max_length=max_source_length,
                         truncation=True,
                         return_tensors="pt",
                         return_offsets_mapping=True)

    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding['attention_mask'].to(device)
    input_ids = torch.tensor(input_ids, dtype=torch.long).cuda()
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()
    labels = torch.tensor(labels, dtype=torch.long).cuda()
    return input_ids, attention_mask, labels

def subwordid_to_text(batch_example, spans_predict, token_idx_maps, results, golds_answers):
    for sample, spans_p, token_idx_map in zip(batch_example, spans_predict, token_idx_maps):

        context = sample['context']
        id = sample['id']
        answers_item = []
        for beg, end in spans_p:
            word_idx_beg, _ = token_idx_map[beg]
            _, word_idx_end = token_idx_map[end]
            # print(word_idx_beg, word_idx_end)
            answer = context[word_idx_beg: word_idx_end]
            # assert answer != ""
            answer = answer.strip()
            if answer == "":
                continue
            answers_item.append(answer)
        # print(answers_item)
        results[id] = answers_item
        golds_answers[id] = sample['answers']
        # print(sample['answers'])
        # print(answers_item)
        # print('-----------')

@torch.no_grad()
def evaluate(model, test_examples, max_len):
    model.eval()
    golds_answers, results = {}, []
    step_trange = tqdm(test_examples)
    acc, count = 0, 0
    for batch_example in step_trange:
        input_ids, attention_mask, labels = get_input_feature(
            batch_example, max_source_length=max_len)
        preds = model(input_ids, attention_mask)
        labels = labels.tolist()
        for label, pred in zip(labels, preds):
            if label == pred:
                acc += 1
            count += 1
            results.append({
                'label': label,
                'pred': pred
            })
    return round(acc / count * 100, 2), results

def read_c3(path):
    dataset = read_dataset(path)
    dataset_new = []
    q_id = 0
    for sample in dataset:
        context = sample[0]
        questions = sample[1]
        for item in questions:
            question = item['question']
            choice = item['choice']
            while len(choice) < 4:
                choice.append('这是一个错误选项')
            answer = item['answer']
            answer_idx = choice.index(answer)
            q_id += 1
            dataset_new.append({
                'id': q_id,
                'question': question,
                'choice': choice,
                'context': context,
                'answer': answer_idx
            })
    return dataset_new



def read_msqa(path):
    dataset = read_dataset(path)
    dataset_new = []
    for sample in dataset:
        id = sample['id']
        question = sample['question']
        context = sample['context']
        question = ['Question', ':'] + question

        context = question + ['Context', ':'] + context
        label = sample['label']
        label = ['O'] * (len(question) + 2) + label
        answers_word_idx = get_entities(label, context)
        answers_word_idx = sorted(answers_word_idx, key=lambda x: x[1])
        answers = [item[0] for item in answers_word_idx]
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
        for ans, beg_idx, end_idx in answers_word_idx:
            assert context_char[context_char_idx_beg[beg_idx]: context_char_idx_end[end_idx]] == ans
            answers_idx_char.append([
                context_char_idx_beg[beg_idx],
                context_char_idx_end[end_idx],
            ])

        dataset_new.append({
            'id': id,
            'context': context_char,
            'answers': answers,
            'answers_idx': answers_idx_char
        })
    return dataset_new

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",
                        default='0',
                        type=str)
    parser.add_argument("--model_name",
                        default='Qwen/Qwen2.5-0.5B',
                        type=str)
    parser.add_argument("--dataset_name",
                        default='c3-m',
                        type=str)
    parser.add_argument("--dataset_split",
                        default='in_house',
                        type=str)
    parser.add_argument("--vanilla",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--only_eval",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--debug",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--results_save_path",
                        default='./results/',
                        type=str)
    parser.add_argument("--save_name",
                        default='MC',
                        type=str)
    parser.add_argument("--train_batch_size",
                        default=24,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--train_micro_batch_size_per_gpu',
                        type=int,
                        default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--output_dir",
                        default='./outputs/',
                        type=str,
                        help="The output dreader2ctory whretriever the model checkpoints will be written.")
    parser.add_argument("--init_checkpoint",
                        default=False,
                        type=ast.literal_eval,
                        help="Initial checkpoint (usually from a pre-trained BERT model)")
    parser.add_argument("--max_len",
                        default=512,
                        type=int)
    parser.add_argument("--wo_auxilary_loss",
                        default=False,
                        type=ast.literal_eval,
                        help="Initial checkpoint (usually from a pre-trained BERT model)")
    parser.add_argument("--gate_model",
                        default="zero_init",
                        # default="norm_init",
                        # default="sigmoid",
                        type=str)
    parser.add_argument("--lr",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epoch_num",
                        default=6,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="random seed for initialization")
    parser.add_argument(
        "--local_rank",
        type=int, default=0
    )
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    vanilla = args.vanilla
    only_eval = args.only_eval
    debug = args.debug
    if args.model_name.endswith('/'):
        args.model_name = args.model_name[:-1]
    model_name_abb = args.model_name.split('/')[-1]
    config_name = f'{args.dataset_name}/{model_name_abb}/'
    dataset_name = args.dataset_name
    save_name = args.save_name

    path_prefix = '.'
    if 'bert' in str(model_name_abb).lower():
        path_save_result = f'{path_prefix}/results/{dataset_name}/encoder-only/{model_name_abb}/'
        output_model_path = f'{path_prefix}/outputs/{dataset_name}/encoder-only/{model_name_abb}/'
    else:
        if vanilla:
            save_name = 'vanilla'
        else:
            save_name = 'enc2dec'
        path_save_result = f'{path_prefix}/results/{dataset_name}/{save_name}/{model_name_abb}/'
        output_model_path = f'{path_prefix}/outputs/{dataset_name}/{save_name}/{model_name_abb}/'
    data_path_base = f'./datas/{args.dataset_name}/'
    data_path_train = f'{data_path_base}/train.json'
    data_path_dev = f'{data_path_base}/dev.json'
    data_path_test = f'{data_path_base}/test.json'

    os.makedirs(path_save_result, exist_ok=True)
    set_seed(args.seed)


    train_examples = read_c3(data_path_train)
    dev_examples = read_c3(data_path_dev)
    test_examples = read_c3(data_path_test)
    if debug:
        train_examples = train_examples[:20]
        dev_examples = dev_examples[:20]
        test_examples = test_examples[:20]

    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    lr = args.lr
    train_micro_batch_size_per_gpu = args.train_micro_batch_size_per_gpu
    gpu_num = torch.cuda.device_count()
    gradient_accumulation = train_batch_size // (train_micro_batch_size_per_gpu * gpu_num)
    assert train_micro_batch_size_per_gpu * gpu_num * gradient_accumulation == train_batch_size
    ds_config = get_train_ds_config(train_batch_size, train_micro_batch_size_per_gpu, lr, gradient_accumulation)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if 'llama' in args.model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    model = Dec2Enc(args.model_name, vanilla).to(device)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()
    model, optimizer, _, __ = deepspeed.initialize(
        config=ds_config,
        model=model,
        model_parameters=parameters,
        training_data=None)

    train_examples = MyDataset(train_examples)
    dev_examples = MyDataset(dev_examples)
    test_examples = MyDataset(test_examples)

    train_sampler = DistributedSampler(train_examples)
    dev_sampler = SequentialSampler(dev_examples)
    test_sampler = SequentialSampler(test_examples)

    train_set = torch.utils.data.DataLoader(
        dataset=train_examples,
        batch_size=train_micro_batch_size_per_gpu,
        sampler=train_sampler,
        collate_fn=lambda x: x
    )

    dev_set = torch.utils.data.DataLoader(
        dataset=dev_examples,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=dev_sampler,
        num_workers=1,
        drop_last=False,
        collate_fn=lambda x: x)

    test_set = torch.utils.data.DataLoader(
        dataset=test_examples,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=1,
        drop_last=False,
        collate_fn=lambda x: x)

    global_rank = torch.distributed.get_rank()
    print('# parameters:', sum(param.numel() for param in model.parameters()))
    print(json.dumps({"lr": args.lr, "model": args.model_name, "seed": args.seed,
                      "bs": args.train_batch_size,
                      "vanilla": vanilla,
                      "epoch": args.epoch_num,
                      "train_path": data_path_train,
                      "dev_path": data_path_dev,
                      "test_path": data_path_test,
                      "train_size": len(train_examples),
                      "dev_size": len(dev_examples),
                      "test_size": len(test_examples),
                      'max_len': args.max_len,
                      'output_model_path': output_model_path,
                      'path_save_result': path_save_result,
                      'init_checkpoint': args.init_checkpoint}, indent=2))

    if args.init_checkpoint:
        init_checkpoint = f'{output_model_path}/pytorch_model.bin'
        checkpoint = torch.load(init_checkpoint, map_location='cpu')
        model_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict()
        for k in list(model_dict.keys()):
            name = k
            if k.startswith('module.bert.bert.'):
                name = k.replace("module.bert.", "")
            new_state_dict[name] = model_dict[k]
            del model_dict[k]
        model.load_state_dict(new_state_dict, False)
        print('init from:', init_checkpoint)

    if only_eval:
        result_score_dev, results_dev = evaluate(model, dev_set, args.max_len)
        print('result_score_dev:', result_score_dev)
        save_dataset(path_save_result + '/dev.json', results_dev)

        result_score_test, results_test = evaluate(model, test_set, args.max_len)
        print('result_score_test:', result_score_test)
        save_dataset(path_save_result + '/test.json', results_test)
        print('save in ', path_save_result)
        exit(0)

    warm_up_ratio = 0.05
    if args.init_checkpoint:
        result_score_dev, results_dev = evaluate(model, dev_set, args.max_len)
        print('best_dev_result:', result_score_dev)
        best_dev_acc = result_score_dev['overlap_f1'] + result_score_dev['em_f1']
    else:
        best_dev_acc = 0

    # for layer in model.model.model.layers:
    #     layer.self_attn.set_fully_attention(False)
    best_dev_result, best_test_result = None, None
    for epoch in range(args.epoch_num):
        tr_loss, nb_tr_steps = 0, 0.1
        step_trange = tqdm(train_set)
        count = 0
        for batch_example in step_trange:

            input_ids, attention_mask, labels = get_input_feature(
                batch_example, max_source_length=args.max_len)

            loss = model(input_ids, attention_mask, labels)
            loss = loss.mean()
            tr_loss += loss.item()
            nb_tr_steps += 1
            model.backward(loss)
            model.step()

            loss_show = ' Epoch:' + str(epoch) + " loss:" + str(
                round(tr_loss / nb_tr_steps, 4)) #+ f" lr:{'%.2E' % scheduler.get_last_lr()[0]}"
            step_trange.set_postfix_str(loss_show)
            if count > 0 and count % 500 == 0:
                f1, results_dev = evaluate(model, dev_set, args.max_len)
                print(f1)
                if f1 >= best_dev_acc:
                    early_stop = 0
                    best_dev_result = f1
                    best_dev_acc = f1
                    save_model(output_model_path, model, optimizer)
                    save_dataset(path_save_result + '/dev.json', results_dev)
                    print('save new best')
                    result_score_test, results_test = evaluate(model, test_set, args.max_len)
                    best_test_result = result_score_test
                    print('test:', result_score_test)
                    save_dataset(path_save_result + '/test.json', results_test)
            count += 1

        f1, results_dev = evaluate(model, dev_set, args.max_len)
        print(f1)
        if f1 >= best_dev_acc:
            early_stop = 0
            best_dev_result = f1
            best_dev_acc = f1
            save_model(output_model_path, model, optimizer)
            save_dataset(path_save_result + '/dev.json', results_dev)
            print('save new best')
            result_score_test, results_test = evaluate(model, test_set, args.max_len)
            best_test_result = result_score_test
            print('test:', result_score_test)
            save_dataset(path_save_result + '/test.json', results_test)

    print('best_dev_result:', best_dev_result)
    print('best_test_result:', best_test_result)
    print(path_save_result)

