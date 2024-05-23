#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import time
import math
import os, sys
import json
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor, device, dtype, nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.set_printoptions(threshold=100000)

import numpy as np

from gpu import (
    add_gpu_params, 
    parse_gpu, 
    distributed_opt, 
    distributed_gather, 
    distributed_sync, 
    cleanup
)

from exp_utils import create_exp_dir

from data_utils import FT_Dataset 
from model import GPT2Config, GPT2LMModel


parser = argparse.ArgumentParser(description='PyTorch GPT2 beam decoding')

add_gpu_params(parser)

parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')

parser.add_argument('--batch_size', type=int, default=10,
                    help='batch size')

parser.add_argument('--seq_len', type=int, default=512,
                    help='number of tokens to predict')

parser.add_argument('--eval_len', type=int, default=256,
                    help='evaluation length')

parser.add_argument('--min_length', type=int, default=0,
                    help='minimum generation length')

parser.add_argument('--model_card', default='gpt2.sm', choices=['gpt2.sm', 'gpt2.md', 'gpt2.lg'],
                    help='model names')

parser.add_argument('--init_checkpoint', default=None, type=str, help='initial checkpoint')

parser.add_argument('--lora_dim', type=int, default=0, help='lora attn dimension')

parser.add_argument('--lora_alpha', type=int, default=128, help='lora attn alpha')

parser.add_argument('--work_dir', type=str, default=os.getenv('PT_OUTPUT_DIR', 'gpt2_model'), 
                    help='working folder')

parser.add_argument('--beam', type=int, default=1, help='beam search size')

parser.add_argument('--length_penalty', type=float, default=1.0, help='length penalty')

parser.add_argument('--no_repeat_ngram_size', type=int, default=4, help='no_repeat_ngram_size')

parser.add_argument('--repetition_penalty', type=float, default=1.0, help='repetition_penalty')

parser.add_argument('--eos_token_id', action='append', type=int, default=[50256], 
                    help='eos token id')

parser.add_argument('--output_file', type=str, default='beam_prediction.jsonl', 
                    help='output file name')


# 打印所有参数结果
def print_args(args):
    if args.rank == 0:
        print('=' * 100)
        for k, v in args.__dict__.items():
            print('        - {} : {}'.format(k, v))
        print('=' * 100)


# 该函数作用十分巨大，在原本的完成了第 n-1 个 token 推理后的 past(n-1) 中的不同 batch 中的 beams(n-1) 中 beam 的分布取决于第 n-1 个 token 在 beams(n-2) 中的分布，同理基于第 n 个 token 更新 past(n-1) 到 past(n)，在第三维重采样
def _reorder_cache(past: Tuple, beam_idx: Tensor) -> Tuple[Tensor]:    # past - (n_layer, 2, batchsize*num_beams, head, max_seq_length, head_features)   beam_idx - (batchsize*num_beams,)
    return tuple(layer_past.index_select(1, beam_idx).contiguous().detach() for layer_past in past)               # (n_layer, 2, batchsize*num_beams, head, max_seq_length, head_features)


# 判断哪些字符需要被惩罚，判断依据是当前字符与紧挨着的前面的 no_repeat_ngram_size-1 个字符构成的长度为 no_repeat_ngram_size 的字符组在前面已经预测的句子中是否已经出现过
def _calc_banned_ngram_tokens(
    prev_input_ids: Tensor,      # (batch_size*num_beams, i)  batch_size*num_beams 个预测结果的完整记录，每个预测结果都是一个句子，句子不一定已完结，初始是 None
    num_hypos: int,              # batch_size * num_beams
    no_repeat_ngram_size: int,   # int
    cur_len: int                 # 迭代推理中的第 cur_len 次推理
) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:                    # 如果当前迭代预测的长度还不足 no_repeat_ngram_size，则无需惩罚
        return [[] for _ in range(num_hypos)]                 # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
    
    generated_ngrams = [{} for _ in range(num_hypos)]         # 针对每个 beam(或推理结果) 预设一个 dict 用于存储对应预测结果的，所有的长度为 no_repeat_ngram_size 的字符组合，键是长度为 no_repeat_ngram_size-1 的元组，值所有已出现的最后一个值
    for idx in range(num_hypos):                              # 遍历每个 beam
        gen_tokens = prev_input_ids[idx].tolist()             # 将第 idx 个 beam 的预测结果转换为列表结构
        generated_ngram = generated_ngrams[idx]               # 拿到预设的第 idx 个 beam 所对应的 dict
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):                        # 遍历第 idx 个 beam 对应的预测结果中所有的 [长度为 no_repeat_ngram_size 的组合]
            prev_ngram_tuple = tuple(ngram[:-1])                                                         # 将上面每个组合的前 no_repeat_ngram_size-1 个字符提取出来组成元组结构，元组可以作为 dict 的 key，而 list 不可以作为 dict 的 key
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]  # 向第 idx 个 beam 所对应的 dict 中填写对应的长度为 no_repeat_ngram_size 的组合，键是元组，值是所有已出现的最后一个字符构成的 list

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())                          # 找到第 hypo_idx 个 beam 当前预测 token 的前 no_repeat_ngram_size-1 个 token 并组成元组结构
        return generated_ngrams[hypo_idx].get(ngram_idx, [])                                             # 从第 hypo_idx 个 beam 对应的 dict 中寻找是否已经存在上一步所得到的元组，如果存在，则其对应的 list 就是需要惩罚的 token 列表

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]                   # 找到每一个 beam 所对应的需要惩罚的 token 列表
    return banned_tokens


# 重复字符惩罚，对所有预测结果中所有已出现过的字符进行惩罚  repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
def _enforce_repetition_penalty_(
    lprobs,               # (batchsize*num_beams, vocabsize) 模型的输出结果
    batch_size, 
    num_beams, 
    prev_output_tokens,   # (batch_size*num_beams, i)  batch_size*num_beams 个预测结果的完整记录，每个预测结果都是一个句子，句子不一定已完结，初始是 None
    repetition_penalty
):
    for i in range(batch_size * num_beams):                                  # 遍历 batch_size*num_beams 个预测结果
        print('prev_output_tokens.shape', prev_output_tokens.shape)
        print('prev_output_tokens[i].shape', prev_output_tokens[i].shape)
        for previous_token in set(prev_output_tokens[i].tolist()):           # 对将第 i 个预测结果所有已出现过的字符进行惩罚
            if lprobs[i, previous_token] < 0:                                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty


# 推理测试时，对每次的模型输出结果进行后处理，包括重复字符惩罚、限制最短长度和重复字符组去除
def _postprocess_next_token_scores(
    scores,                   # (batchsize*num_beams, vocabsize) 模型的输出结果
    history,                  # (batch_size*num_beams, i)  batch_size*num_beams 个预测结果的完整记录，每个预测结果都是一个句子，句子不一定已完结，初始是 None
    cur_len,                  # 预测到第几个字符
    batch_size,               # 当前 1
    num_beams,                # 当前 10
    repetition_penalty=1.0,   # 当前 1.0                               
    no_repeat_ngram_size=4,   # 当前 4
    bad_words_ids=None,
    min_length=0,             # 当前 0
    max_length=100,
    eos_token_id=None,        # 当前 628

):  # 1. 重复字符惩罚，对所有预测结果中所有已出现过的字符进行惩罚  repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0 and history is not None:
        _enforce_repetition_penalty_(scores, batch_size, num_beams, history, repetition_penalty)

    # 2. 限制最短长度，如果当前预测的长度还不足 min_length，则不允许预测出 eos 这个终止字符
    # score: batch_size * beam, vocab
    # set eos token prob to zero if min_length is not reached
    if eos_token_id is not None and cur_len < min_length:
        for eos in eos_token_id:
            scores[:, eos] = -float("inf")

    # 3. 重复字符组去除，在整个预测的句子中不允许出现连续大于等于 no_repeat_ngram_size 个重复的字符组，例如这个参数设置为 2，那如果之前句子中出现了 New York，那以后的预测中不允许再出现这两个连续的字符 New York
    if no_repeat_ngram_size > 0 and history is not None:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        # 判断哪些字符需要被惩罚，判断依据是当前字符与紧挨着的前面的 no_repeat_ngram_size-1 个字符构成的长度为 no_repeat_ngram_size 的字符组在前面已经预测的句子中是否已经出现过
        banned_batch_tokens = _calc_banned_ngram_tokens(history, num_batch_hypotheses, no_repeat_ngram_size, cur_len)   # (batchsize*num_beams,)
        for i, banned_tokens in enumerate(banned_batch_tokens):                                                         # banned_tokens 是个 list
            scores[i, banned_tokens] = -float("inf")

    return scores


# 一个 batch 中包含 batchsize 个样本，一个样本包含 num_beams 个 beam，一个 beam 对应一个句子(token列表)，最终要求一个样本输出一个最佳句子，模型每完成一次推理和后处理后都会产生 batch_size*num_beams 个预测结果
# 该函数的作用是从模型的预测输出结果中挑选出合适的已预测完结的句子，并将这个句子从迭代预测流程中删除，并让新的句子补充进迭代流程中继续迭代预测，解决了【如果一个句子已经预测输出 EOS 终止符后是否还需继续参与迭代预测流程的问题】
# 删除句子的方式是将其总得分设置为 -inf，然后下一轮迭代不管再预测出一个什么 token，这个句子的总得分还是 -inf，然后当基于 beam search 中的 topK 选取 num_beams 个 beam 时，就会自动删除这个句子
# 从模型的预测输出结果中挑选出的合适的已预测完结的句子，需要与这个句子对应样本当前的最佳句子最对比，如果新句子得分高，则保留新句子，否则不修改原有保留的旧句子，结果多次调用，最终该函数会输出每个样本的最佳输出句子和对应得分
def _add_beam_candidate(
    best_score,           # dict 记录一个 batch 中每个样本当前最佳输出句子的总得分，会随着迭代预测不断变化，这个类似于求一个列表极小值编程时的记录当前最小值的中间变量，辅助 best_sequence 实现每个样本最佳输出句子的获取
    best_sequence,        # (batch_size, args.eval_len) 用于记录一个 batch 中每个样本的最佳输出句子，每个样本的最佳输出句子是 不断迭代过程中 这个样本的所有beam中 达到EOS终止的句子中 得分最大的句子
    batch_size,           # 一个 batch 所含有的样本数量
    num_beams,            # 一个样本所含有的 beam 数量
    beam_scores,          # (batchsize, num_beams)     batch_size*num_beams 个预测结果的每个句子当前得分
    history,              # (batch_size*num_beams, i)  batch_size*num_beams 个预测结果的完整记录，每个预测结果都是一个句子，句子不一定已完结
    eos_token_id=None     # 628
):
    last_tokens = history[:, -1]                                                 # (batch_size*num_beams,)   batch_size*num_beams 个预测结果的最新 token
    for _i in range(batch_size * num_beams):                                     # 遍历 batch_size*num_beams 中的每个预测结果，模型每完成一次推理和后处理后都会产生 batch_size*num_beams 个预测结果
        if eos_token_id is None or last_tokens[_i] in eos_token_id:              # 如果第 _i 个预测结果的最新一个 token 是终止字符，即一个句子完成预测了，无需继续预测了，通过设置 -inf，匹配 top-k 策略会从 beams 中删除这个句子
            cur_len = history.shape[-1]                                          # 所有预测结果的当前句子(即token列表)长度
            _score = beam_scores.view(-1)[_i] / cur_len ** args.length_penalty   # 对第 _i 个预测结果的总得分进行惩罚，通过 length_penalty 调整最终输出结果偏向于短句子还是长句子，大于 1 偏向于短句子，反之则偏向长句子

            batch_id = _i // num_beams                                           # 计算 batch_size*num_beams 中第 _i 个预测结果隶属于 batch 中的第几个样本
                                                                                 # 至此
            if not batch_id in best_score or best_score[batch_id] < _score:      # 完成了一个句子的预测，并获取了这个句子在 batch 中所属的样本和总得分，然后与之前该样本最佳句子的得分对比，得分更高则保留最新的句子，否则不保留
                best_score[batch_id] = _score                                    # 更新 batch 中第 batch_id 样本的最佳得分，每个样本会预测多个句子，挑选得分最高的句子保留
                best_sequence[batch_id][:cur_len] = history[_i]                  # 更新 batch 中第 batch_id 样本的最佳句子

            beam_scores.view(-1)[_i] = -float("inf")                             # 如果第 _i 个预测结果的最新一个 token 是终止字符，则设置其所对应的句子总得分为 -inf，这样在 beam search 的下一轮 topK 操作时会删除这个句子
    # best_score     最终输出结果，每个样本的最佳输出句子的得分
    # best_sequence  最终输出结果，每个样本的最佳输出句子


# 完成所有输入数据的推理，并基于 beam search 进行结果处理，并存储最终结果，最终结果是一个字典，键是 id 编号，值是一个子 dict，子 dict 的键是“id”和“predict”，predict 的值是最终的每个样本的最佳预测结果
def beam(model, data_iter, args):
    model.eval()
    total_loss = 0.
    start_time = time.time()

    all_predictions = {}                                       # 用于存储最终的预测结果，键是 id 编号，值是一个子 dict，子 dict 的键是“id”和“predict”，predict 的值是最终的每个样本的最佳预测结果
    with torch.no_grad():
        for idx, data in enumerate(data_iter):                 # 迭代输入所有 batch
            data = {key: value for key, value in data.items()} # dict

            _id = data['id'].to(args.device)                   # (batchsize,)                 此 id 即为该样本在数据集中对应的行数
            _query = data['query'].to(args.device)             # (batchsize, max_seq_length)  仅包含数据集中的 input 部分，最长 max_seq_length，不足的补充 0
            _query_len = data['query_len'].to(args.device)     # (batchsize,)                 对应 input 的实际长度

            ## local adaptation start.

            ## local adaptation end.

            output = None
            score = None

            batch_size = _id.size(0)
            num_beams = args.beam
            length_penalty = args.length_penalty

            _batch = torch.arange(0, _id.size(0), device=args.device, dtype=torch.long) # [0, ..., batchsize-1]
            
            past = None
            len_past = None
                                                                                        # 在输入数据的 batch 维度复制 num_beams 份，以实现束搜索
            _query = _query.repeat(1, num_beams).view(batch_size * num_beams, -1)       # (batchsize*num_beams, max_seq_length) <-- (batchsize, max_seq_length*num_beams) <-- (batchsize, max_seq_length)
            _query_len = _query_len.unsqueeze(-1).repeat(1, num_beams).view(-1)         # (batchsize*num_beams,) <-- (batchsize, num_beams) <-- (batchsize, 1) <-- (batchsize,)   [n1, ..., n1, n2, ..., n2, ... ]
            _bbatch = _batch.unsqueeze(-1).repeat(1, num_beams).view(-1)                # (batchsize*num_beams,) <-- (batchsize, num_beams) <-- (batchsize, 1) <-- (batchsize,)   [1, ..., 1, ..., batchsize-1, ..., batchsize-1]
            
            # scores for each sentence in the beam
            beam_scores = torch.zeros(
                (batch_size, num_beams), dtype=torch.float, device=_query.device        # (batchsize, num_beams)       作用是统计每个 beam 中已预测句子的总得分，每迭代预测一次字符都需要基于束搜索原理更新 beam_scores
            )

            best_sequence = torch.zeros(                                                # 每个样本会预测多个句子，挑选得分最高的句子保留
                (batch_size, args.eval_len), dtype=torch.long, device=_query.device     # (batch_size, args.eval_len)  作用是记录每个样本的当前最佳预测句子，每迭代预测一次字符都需要基于束搜索原理更新 best_sequence
            )
            best_score = {}   # dict 记录一个 batch 中每个样本当前最佳输出句子的总得分，会随着迭代预测不断变化，这个类似于求一个列表极小值编程时的记录当前最小值的中间变量，辅助 best_sequence 实现每个样本最佳输出句子的获取

            history = None
            with torch.no_grad():
                for i in range(0, args.eval_len):                                       # 对一个 batch 数据开始迭代推理，迭代中第一次把 input+补长字符0 送入模型，第二次开始每次只送入单个字符

                    ## 1. 进行模型推理
                    if i == 0:                                                          # 推理测试时，第一次输入为数据集中完整 input，具体来说是 input+补长字符0，长度为 max_seq_length，因此 past 的序列固定为 max_seq_length
                        logits, past = model(_query)                                    # (batchsize*num_beams, max_seq_length, vocabsize) <-- (batchsize*num_beams, max_seq_length)，past 尺寸见下面，尺寸始终不变
                        logits = logits[_bbatch, (_query_len-1).long(), :]              # (batchsize*num_beams, vocabsize) 这个地方怀疑有 bug，如果 batch=1 则不会触发这个 bug，正确是 logits[:, (_query_len-1).long(), :] 
                    else:                                                               # 或者，如果 logit 的尺寸是 (batchsize, max_seq_length, vocabsize)，那写成 [_bbatch, (_query_len-1).long(), :] 也是可以的!!!!!!!!!!
                        #print('token_id.shape', token_id.shape, token_id)
                        #print('past.shape', past[0].shape)                             # 推理测试时，从第二次开始输入变成上一次输出的单个字符，但需要缓存之前所有字符的 k v 以避免重复计算
                        #print('len_past.shape', len_past.shape, len_past)              # past 为一个列表，长度 n_layer，里面每个 tensor 大小 (2, batchsize*num_beams, head, max_seq_length, head_features)，第一维是 k v 区分
                                                                                        # len_past 为一个列表，长度是 n_layer，里面每个 tensor 大小是 1，用于描述 past 中对应 tensor 的有效长度
                        logits, past = model(token_id, past=past, len_past=len_past)    # (batchsize*num_beams, q_seq_length, vocabsize), (n_layer, 2, batchsize*num_beams, head, max_seq_length, head_features)
                        logits = logits[:, -1, :]                                       # (batchsize*num_beams, vocabsize)  拿到末尾一个字符，作为本轮迭代的输出

                    ## 2. 对模型推理结果进行后处理
                    logits = _postprocess_next_token_scores(                            # 推理测试时，对每次的输出结果进行后处理，包括重复字符惩罚、限制最短长度和重复字符组去除
                        logits,                                                         # (batchsize*num_beams, vocabsize) 模型的输出结果
                        history,                                                        # (batch_size*num_beams, i)  batch_size*num_beams 个预测结果的完整记录，每个预测结果都是一个句子，句子不一定已完结，初始是 None
                        i,                                                              # 预测到第几个字符
                        batch_size,
                        num_beams,
                        repetition_penalty=args.repetition_penalty,                                
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        min_length=args.min_length,
                        eos_token_id=args.eos_token_id,
                    )

                    softmax_probs = F.softmax(logits, dim=-1)                           # (batchsize*num_beams, vocabsize)   softmax(out)       通常意义上的预测概率分布
                    #_prob, _w_idx = torch.topk(softmax_probs, num_beams) # batch_size, beam
                    vocab_size = softmax_probs.shape[-1] 
                    
                    ## 3. 进行束搜索 beam search，整理每个预测结果的得分和 topK 选取
                    _logprob = torch.log(softmax_probs)                                 # (batchsize*num_beams, vocabsize)   log(softmax(out))  对概率分布取 log 变负值，只是改变了相对差距，放大偏小的值，但是不改变大小排序
                    if i == 0:
                        next_scores = _logprob.view(batch_size, num_beams, -1)[:, 0, :] # (batchsize, vocabsize)                                 第一次预测不存在 beams 概念，选择第 0 个样本，所以每个样本只有 vocabsize 个预测结果
                    else:
                        next_scores = beam_scores.unsqueeze(-1) + _logprob.view(batch_size, num_beams, -1)  # (batchsize, num_beams, vocabsize)  第二次预测开始存在 beams 所以每个样本有 num_beams*vocabsize 个预测结果
                        next_scores = next_scores.view(batch_size, -1)                                      # (batchsize, num_beams*vocabsize)

                    next_scores, next_tokens = torch.topk(                                                  # (batchsize, num_beams)，(batchsize, num_beams)  前者是得分，后者是 num_beams*vocabsize 中的总索引
                        next_scores, num_beams, dim=1, largest=True, sorted=True                            # 从 num_beams*vocabsize 中挑选 beam 个 topk 结果，num_beams*vocabsize 来自同一个 batch，第一次预测则例外
                    )
                    
                    ## 4. 对束搜索结果进行后处理，主要是更新各种变量，以备下一轮的模型推理和束搜索，步骤 1-4 可以循环往复执行，不依赖于步骤 5 6
                    beam_id = (next_tokens // vocab_size).view(-1)                                          # batch_size*num_beams       束索引，即在 num_beams 中的索引
                    token_id = (next_tokens % vocab_size).view(-1).unsqueeze(-1)                            # (batch_size*num_beams, 1)  在每束中的 token 索引，即在 vocabsize 中的索引

                    beam_idx = beam_id.view(batch_size, num_beams) + (_batch * num_beams).unsqueeze(-1)     # (batchsize, num_beams) <-- (batchsize, num_beams) + (batchsize, 1) 在 batch_size*num_beams 中的总体束索引
                    past = _reorder_cache(past, beam_idx.view(-1))                                          # (n_layer, 2, batchsize*num_beams, head, max_seq_length, head_features) 其实就是基于总体束索引结果对第三维进行重采样
                    beam_scores = next_scores                                                               # (batchsize, num_beams)
                    len_past = (_query_len + i).long()                                                      # (n_layer, batchsize*num_beams)  past 中 k v 的有效长度

                    if history is None:                                                                       # batch_size*num_beams 个预测结果的完整记录，每个预测结果都是一个句子，句子不一定已完结，初始是 None
                        history = token_id.detach()                                                           # (batch_size*num_beams, 1)
                    else:
                        history = torch.cat((history[beam_idx.view(-1)], token_id.detach()), dim=1).detach()  # (batch_size*num_beams, i)

                    ## 5. 每迭代一次，都尝试从束搜索的结果中寻找最佳预测句子结果，并不保证每次都可以找到，寻找这个动作的触发是预测到一个终止字符 EOS
                    _add_beam_candidate(
                        best_score, best_sequence, batch_size, num_beams, beam_scores, history, 
                        eos_token_id=args.eos_token_id
                    )
                
                ## 6. 在一个样本的迭代推理达到最长推理长度后，不管有没有预测到终止字符 EOS，都整体更新一遍每个样本的最佳预测句子结果，每个样本只得到一个最佳预测结果
                _add_beam_candidate(
                    best_score, best_sequence, batch_size, num_beams, beam_scores, history
                )

            # 对分布在不同 GPU 上的 batch 进行合并，每个 GPU 都会单独运行一个 batch，每个 batch 的大小就是预设的 batchsize
            with torch.no_grad():
                _id = distributed_gather(args, _id)                           # (n_GPU*batchsize,)
                output = distributed_gather(args, best_sequence)              # (n_GPU*batchsize, eval_len)
                #score = distributed_gather(args, score)
                distributed_sync(args)

            # 整理所有 GPU 单个 batch 的预测结果
            if args.rank == 0:
                _id = _id.view(-1).cpu()                                      # (n_GPU*batchsize,)
                output = output.view(-1, output.shape[-1]).cpu()              # (n_GPU*batchsize, eval_len)
                #score = score.view(-1, score.shape[-1]).cpu()

                for _b in range(0, _id.shape[-1]):
                    _i = int(_id[_b].item())
                    all_predictions[_i] = {}
                    all_predictions[_i]['id'] = _i                            # 整数，是输入样本在数据集中的行数
                    all_predictions[_i]['predict'] = output[_b].tolist()      # 长度为 eval_len 的 list，每个数都是 token id
                    #all_predictions[_i]['score'] = score[_b].tolist()

                if idx % 10 == 0:
                    print('inference samples', idx)

    # 存储最终的所有 batch 的预测结果，预测结果是一个字典，键是 id 编号，值是一个子 dict，子 dict 的键是“id”和“predict”，predict 的值是最终的每个样本的最佳预测结果，这个结果是长度为 eval_len 的 list，每个数都是 token id
    if args.rank == 0:
        pred_file = os.path.join(args.work_dir, args.output_file) 
        print('saving prediction file', pred_file)
        with open(pred_file, 'w') as writer:
            for _i in all_predictions:
                writer.write(json.dumps(all_predictions[_i]) + '\n')
    

if __name__ == '__main__':

    # 参数相关
    args = parser.parse_args() # 定义所有参数
    parse_gpu(args)            # 分布式参数设定
    print_args(args)           # 打印所有参数
    
    if args.rank == 0:
        args.logging = create_exp_dir(args.work_dir)      # 创建一个日志记录器，logging 其实就是一个函数，函数的输入就是希望记录下来的字符串，logging 函数的本质就是一个 with open 操作，不断往文本文件中追加行

    # 完成分布式数据集的定义，其中 args.batch_size 为每个节点上都具有这个的 size
    valid_data = FT_Dataset(
        args.data, args.batch_size, args.seq_len, args.eval_len, 
    )    
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)
    valid_loader = DataLoader(
        valid_data, batch_size=args.batch_size, num_workers=0, shuffle=False,    # 分布式数据集的打乱策略再查吧，应该是打乱了，但不是通过 shuffle 进行打乱
        pin_memory=False, drop_last=False, sampler=valid_sampler
    )

    # 完成 GPT2 模型定义
    if args.model_card == 'gpt2.sm':
        config = GPT2Config(
            n_embd=768, n_layer=12, n_head=12, 
            lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha,
        )
    elif args.model_card == 'gpt2.md':
        config = GPT2Config(
            n_embd=1024, n_layer=24, n_head=16, 
            lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha,
        )
    elif args.model_card == 'gpt2.lg':
        config = GPT2Config(
            n_embd=1280, n_layer=36, n_head=20, 
            lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha,
        )

    # 完成模型生成和初始化
    lm_net = GPT2LMModel(config)
    if args.init_checkpoint is not None:
        print('loading model pretrained weight.')
        cp = torch.load(args.init_checkpoint, map_location=torch.device('cpu'))
        lm_net.load_weight(cp)    
    lm_net = lm_net.cuda()

    print('model sampling ...')
    beam(lm_net, valid_loader, args) # 完成所有输入数据的推理，并基于 beam search 进行结果处理，并存储最终结果，最终结果是一个字典，键是 id 编号，值是一个子 dict，子 dict 的键是“id”和“predict”
    distributed_sync(args)
    print('cleanup dist ...')
    cleanup(args)