# coding=utf-8
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizerFast, AlbertTokenizerFast, DebertaTokenizerFast, AutoModel, get_linear_schedule_with_warmup, AutoConfig
import os
import random
import torch
from utils import save_dataset, set_seed, save_model, read_dataset
import json
import argparse
from torch import nn
import math
from collections import OrderedDict
import copy
import ast
import numpy as np
from deepspeed_config import get_train_ds_config
import deepspeed
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from eval_mlqa import evaluate_squad_mlqa

device = torch.device("cuda:0")

class MyDataset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return sample

class Dec2Enc(nn.Module):
    def __init__(self, model_path, vanilla):
        super(Dec2Enc, self).__init__()
        config = AutoConfig.from_pretrained(model_path)
        config._attn_implementation = 'eager'
        self.model_path = model_path
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

        self. linear = nn.Linear(self.model.config.hidden_size, 2)

    def forward(self, input_ids,  attention_mask, labels=None):

        if 'bert' in self.model_path.lower() or 'xlm' in self.model_path.lower():
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 # attn_implementation="eager",
                                 output_hidden_states=True,
                                 # use_cache=False,
                                 output_attentions=True,
                                 return_dict=True)
            sequence_output = outputs.hidden_states[-1]
        else:
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 # attn_implementation="eager",
                                 output_hidden_states=True,
                                 use_cache=False,
                                 output_attentions=True,
                                 return_dict=True)
            sequence_output = outputs.hidden_states[-1]
        logits = self.linear(sequence_output)

        if labels is not None:
            loss = 0
            for i in range(2):
                logits_item = logits[:, :, i]
                labels_i = labels[:, :, i]
                # print(torch.sum(labels_i, dim=1))
                logits_item = torch.clamp(logits_item, -80, 80)
                logits_pos = torch.exp(-logits_item)
                logits_pos = torch.sum(logits_pos * labels_i * attention_mask, dim=-1)
                loss_pos = torch.log(1 + logits_pos)
                logits_neg = torch.exp(logits_item)
                logits_neg = torch.sum(logits_neg * (1 - labels_i) * attention_mask, dim=-1)
                loss_neg = torch.log(1 + logits_neg)
                loss_i = loss_pos + loss_neg
                loss_i = loss_i.mean()
                loss += loss_i
            return loss
        else:
            logits_s = logits[:, :, 0]
            logits_e = logits[:, :, 1]
            logits_s = logits_s.cpu().tolist()
            logits_e = logits_e.cpu().tolist()

            def get_result(logits_s, logits_e):
                max_s = max(logits_s)
                best_beg = logits_s.index(max_s)
                max_e = max(logits_e[best_beg:])
                best_end = logits_e[best_beg:].index(max_e) + best_beg
                return best_beg, best_end

            results = []
            for logits_s_i, logits_e_i in zip(logits_s, logits_e):
                best_beg, best_end = get_result(logits_s_i, logits_e_i)

                results.append([best_beg, best_end])
            return results

def get_input_feature(features, max_source_length):

    input_texts, contexts = [], []
    label_char_idxs_beg, label_char_idxs_end = [], []
    for sample in features:
        context = sample['context']
        contexts.append(context)
        answers_idx = sample['answers_idx']
        answers_idx = sorted(answers_idx, key=lambda x: x[0])
        beg_idxs, end_idxs = [], []
        for beg, end in answers_idx:
            beg_idxs.append(beg)
            end_idxs.append(end)
            # break
        label_char_idxs_beg.append(beg_idxs)
        label_char_idxs_end.append(end_idxs)

    if tokenizer.eos_token is not None:
        contexts = [item + tokenizer.eos_token for item in contexts]
    # tokenizer.add_eos_token = True
    encoding = tokenizer(contexts,
                         padding='longest',
                         max_length=max_source_length,
                         truncation=True,
                         return_tensors="pt",
                         return_offsets_mapping=True)

    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding['attention_mask'].to(device)
    offset_mapping = encoding['offset_mapping']
    offset_mapping = offset_mapping.tolist()

    bs, seq_len = input_ids.size()
    labels = np.zeros([bs, seq_len, 2])

    def label_tokens(offset_mapping, label_char_idxs, dim):
        for b_i, (offset_mapping_item, label_char_idxs_item) in enumerate(
                zip(offset_mapping, label_char_idxs)):
            idx = 0
            for seq_i, (token_beg, token_end) in enumerate(offset_mapping_item):
                if idx >= len(label_char_idxs_item):
                    break
                label_idx = label_char_idxs_item[idx]
                if dim == 1:
                    if token_beg < label_idx and label_idx <= token_end:
                        labels[b_i][seq_i][dim] = 1
                        idx += 1
                else:
                    if token_beg <= label_idx and label_idx < token_end:
                        labels[b_i][seq_i][dim] = 1
                        idx += 1
    label_tokens(offset_mapping, label_char_idxs_beg, 0)
    label_tokens(offset_mapping, label_char_idxs_end, 1)
    input_ids = torch.tensor(input_ids, dtype=torch.long).cuda()
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()
    labels = torch.tensor(labels, dtype=torch.long).cuda()

    return input_ids, attention_mask, labels, offset_mapping, contexts


def subwordid_to_text(batch_example, spans_predict, token_idx_maps, results, golds_answers):
    for sample, [beg, end], token_idx_map in zip(batch_example, spans_predict, token_idx_maps):
        context = sample['context']
        id = sample['id']
        word_idx_beg, _ = token_idx_map[beg]
        _, word_idx_end = token_idx_map[end]
        answer = context[word_idx_beg: word_idx_end]
        answer = answer.strip()
        results[id] = answer#.replace(' ', '')
        golds_answers[id] = sample['answers']


@torch.no_grad()
def evaluate(model, test_examples, max_len, lang):
    model.eval()

    golds_answers, results = {}, {}
    step_trange = tqdm(test_examples)
    for batch_example in step_trange:
        # print(batch_example)
        input_ids, attention_mask, labels, offset_mapping_contexts, contexts = get_input_feature(
            batch_example, max_source_length=max_len)
        spans_predict = model(input_ids, attention_mask, labels=None)
        subwordid_to_text(batch_example, spans_predict, offset_mapping_contexts, results, golds_answers)
    results_cp = {}
    keys = results.keys()
    for key in keys:
        results_cp[key] = results[key]
    results_output = []
    for samples in test_examples:
        for sample in samples:
            id = sample['id']
            results_output.append({
                'id': id,
                'context': sample['context'],
                'answer': golds_answers[id],
                'pred': results[id]
            })

    result_score = evaluate_fun(copy.deepcopy(golds_answers), copy.deepcopy(results),lang   )
    result_score = {
        'em_f1': result_score['exact_match'],
        'overlap_f1': result_score['f1']
    }
    return result_score, results_output

def read_msqa(path):
    dataset = read_dataset(path)
    dataset_new = []
    for article in dataset:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                answers = qa['answers']
                id = qa['id']
                offset_context = f'Question: {question} \n Context: '
                offset = len(offset_context)
                answers_idx_char = [(item['answer_start'] + offset,
                                     item['answer_start'] + len(item['text']) + offset) for item in answers]
                answers = [item['text'] for item in answers]
                offset_context = offset_context + context
                for (beg, end), answer in zip(answers_idx_char, answers):
                    assert offset_context[beg: end] == answer
                dataset_new.append({
                    'id': id,
                    'context': offset_context,
                    'answers': answers,
                    'answers_idx': answers_idx_char
                })
    return dataset_new


def read_msqa_zh(path):
    dataset = read_dataset(path)
    dataset_new = []
    for article in dataset:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            context = ' '.join(list(context))
            for qa in paragraph['qas']:
                question = qa['question']
                answers = qa['answers']

                question = ' '.join(list(question))
                for item in answers:
                    text = item['text']
                    item['text'] = ' '.join(list(text))
                    answer_start = item['answer_start']
                    item['answer_start'] += answer_start



                id = qa['id']
                offset_context = f'Question: {question} \n Context: '
                offset = len(offset_context)
                answers_idx_char = [(item['answer_start'] + offset,
                                     item['answer_start'] + len(item['text']) + offset) for item in answers]
                answers = [item['text'] for item in answers]
                offset_context = offset_context + context
                for (beg, end), answer in zip(answers_idx_char, answers):
                    assert offset_context[beg: end] == answer
                dataset_new.append({
                    'id': id,
                    'context': offset_context,
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
                        default='mlqa',
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
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--train_micro_batch_size_per_gpu',
                        type=int,
                        default=8,
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
                        default=20,
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
    only_eval = args.only_eval
    debug = args.debug
    if args.model_name.endswith('/'):
        args.model_name = args.model_name[:-1]
    model_name_abb = args.model_name.split('/')[-1]
    config_name = f'{args.dataset_name}/{model_name_abb}/'
    dataset_name = args.dataset_name
    vanilla = args.vanilla
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
    read_dataset_fun = read_msqa

    evaluate_fun = evaluate_squad_mlqa

    data_path_train = './datas/mlqa/train.json'
    train_examples = read_dataset_fun(data_path_train)
    random.shuffle(train_examples)
    dev_examples = []
    for lang in ['ar', 'de', 'en',
                 'es', 'hi', 'vi',
                 'zh']:
        data_path_dev = f'./datas/mlqa/dev/dev-context-{lang}-question-{lang}.json'
        dev_examples += read_dataset_fun(data_path_dev)
    if debug:
        train_examples = train_examples[:20]
        dev_examples = dev_examples[:20]

    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    lr = args.lr
    train_micro_batch_size_per_gpu = args.train_micro_batch_size_per_gpu
    gpu_num = torch.cuda.device_count()
    gradient_accumulation = train_batch_size // (train_micro_batch_size_per_gpu * gpu_num)
    assert train_micro_batch_size_per_gpu * gpu_num * gradient_accumulation == train_batch_size
    ds_config = get_train_ds_config(train_batch_size, train_micro_batch_size_per_gpu, lr, gradient_accumulation)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if 'llama' in args.model_name.lower() or 'mistral' in args.model_name.lower():
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

    train_sampler = DistributedSampler(train_examples)
    dev_sampler = SequentialSampler(dev_examples)

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
                      # "test_size": len(test_examples),
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
        for lang in ['ar', 'de', 'en',
                     'es', 'hi', 'vi',
                     'zh']:
            if lang != 'en':
                continue
            data_path_test = f'./datas/mlqa/test/test-context-{lang}-question-{lang}.json'
            test_examples = read_dataset_fun(data_path_test)
            test_examples = MyDataset(test_examples)
            test_sampler = SequentialSampler(test_examples)
            test_set = torch.utils.data.DataLoader(
                dataset=test_examples,
                batch_size=eval_batch_size,
                shuffle=False,
                sampler=test_sampler,
                num_workers=1,
                drop_last=False,
                collate_fn=lambda x: x)

            result_score_test, results_test = evaluate(model, test_set, args.max_len, lang)
            print(f'{lang}:', result_score_test)
            save_dataset(path_save_result + f'/{lang}.json', results_test)
            print('save in ', path_save_result)
        exit(0)

    warm_up_ratio = 0.05
    if args.init_checkpoint:
        result_score_dev, results_dev = evaluate(model, dev_set, args.max_len, lang='en')
        print('best_dev_result:', result_score_dev)
        best_dev_acc = result_score_dev['overlap_f1'] + result_score_dev['em_f1']
    else:
        best_dev_acc = 0

    best_dev_result, best_test_result = None, None
    count = 0
    for epoch in range(args.epoch_num):
        tr_loss, nb_tr_steps = 0, 0.1
        step_trange = tqdm(train_set)
        for batch_example in step_trange:
            count += 1
            input_ids, attention_mask, labels, offset_mapping_contexts, contexts = get_input_feature(
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
            if count  > 0 and count % 1000 == 0:
                result_score_dev, results_dev = evaluate(model, dev_set, args.max_len,lang='en')
                f1 = result_score_dev['overlap_f1'] + result_score_dev['em_f1']
                print(result_score_dev)
                if f1 >= best_dev_acc:
                    early_stop = 0
                    best_dev_result = result_score_dev
                    best_dev_acc = f1
                    save_model(output_model_path, model, optimizer)
                    save_dataset(path_save_result + '/dev.json', results_dev)
                    print('save new best')

        result_score_dev, results_dev = evaluate(model, dev_set, args.max_len, lang='en')
        f1 = result_score_dev['overlap_f1'] + result_score_dev['em_f1']
        print(result_score_dev)
        if f1 >= best_dev_acc:
            early_stop = 0
            best_dev_result = result_score_dev
            best_dev_acc = f1
            save_model(output_model_path, model, optimizer)
            save_dataset(path_save_result + '/dev.json', results_dev)
            print('save new best')

    print('best_dev_result:', best_dev_result)