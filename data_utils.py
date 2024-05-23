#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os, sys
import glob
import random
from collections import Counter, OrderedDict
import numpy as np
import torch
import json

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# 通过 pad_token 字符将 tokens 字符组补足到 max_seq_length 固定长度
def padding_tokens(tokens, max_seq_length, pad_token, direct, max_context_length=0):

    if max_context_length == 0:
        max_context_length = max_seq_length            # max_context_length 为在 max_seq_length 中最长有效长度，如果不指定具体值，则 max_context_length = max_seq_length，这个值用于在推理过程中限制 input 长度，为 output 留下足够空间

    if len(tokens) > max_context_length:
        if direct > 0:
            pad_tokens = tokens[:max_context_length]   # 从前往后截取数据，模型训练时用
        else:
            pad_tokens = tokens[-max_context_length:]  # 从后往前截取数据，推理测试时用
    else:
        pad_tokens = tokens
    token_len = len(pad_tokens)                                                       # tokens 字符组的长度，这是真实语句有效长度
    pad_tokens = pad_tokens + [pad_token for _ in range(max_seq_length - token_len)]  # 通过 pad_token 字符将 tokens 字符组补足到 max_seq_length 固定长度
    return pad_tokens, token_len


# 数据读取基础类，输入数据事先需要完成切词和映射为整数索引值
class FT_Dataset(Dataset):       # 训练时，模型的输入是 input+output+补长字符，输出是向前错一位的输入字符组，测试时，第一次输入是 input+补长字符，第二次开始输入是 input+迭代输出的字符+补长字符，input 和 output 不是指模型输入输出，指的是数据集中的概念
    def __init__(self, ft_file, batch_size, max_seq_length, 
                 max_eval_length=0, joint_lm=False, prefix_len=0, infix_len=0, 
                 prefix_cursor=1000000, infix_cursor=2000000):
        self.ft_file = ft_file                         # 数据集路径，需要是已经经过切词和映射为整数索引值的数据文件，后缀为 jsonl，不是原生文字文件
        self.ft_samples = self.read_ft_file(ft_file)   # 读取数据集，读取后形式为 [[input,output], ...]，已经完成切词和整数索引映射，并在 input 和 output 后面都添加了一个 50256 的终止符，在数据集中 input 或 output 均不等长
        self.batch_size = batch_size                   # batchsize，训练时和测试时，batchsize 均可以不为 1
        self.num_examples = len(self.ft_samples)       # 数据集总长度，即数据集中含有多少 [input,output]
        self.max_seq_length = max_seq_length           # 最大输入长度，最终所有不等长的模型输入数据都会被填充至这个最大输入长度，然后才等长的送入模型，不足的填充【补长字符】，否则无法拼接成 batch
        self.max_eval_length = max_eval_length         # 最大推理长度，训练时用不到，推理时最大第一次有效输入长度为 max_seq_length-max_eval_length，后续一个一个字符推理时最大有效迭代输出长度为 max_eval_length，实际输入模型长度依然是 max_seq_length
        self.rng = random.Random(911)                  # 随机数生成器
        self.joint_lm = joint_lm                       # 训练时，选择哪些输出数据计算 loss，'clm' 模式仅仅选取模型输出中的 output 部分计算 loss, 'jlm' 模式则选取模型输出中的 input+output 部分计算 loss
                                                    
        self.num_batches = int((self.num_examples + self.batch_size - 1) / self.batch_size)   # 一个 epoch 总 step 数量，最后一个 step 中数据量有可能不足，需要随机填充

        self.prefix_len = prefix_len                   # 前缀字符长度
        self.infix_len = infix_len                     # 中缀字符长度
        self.prefix_cursor = prefix_cursor             # 前缀字符
        self.infix_cursor = infix_cursor               # 中缀字符，本次使用中，前缀和中缀均没有使用，但在 input 和 output 结尾处有一个 50256 的终止符，终止符在最后句号后面，这个终止符在 jsonl 文件中就已经有了

    def __len__(self):
        return self.num_batches * self.batch_size      # 总数据量 >= 数据集总长度
        
    def __getitem__(self, item):
        if(item >= self.num_examples):                 # 由于 batch 整除问题，导致数据集总长度小于总数据量，所以当 item 超过数据集总长度时，随机选取一个数据以填充最后一个 batch
            item = self.rng.randint(0, self.num_examples - 1)

        example = self.ft_samples[item]                # 拿到一个数据 [input,output]
        context = example[0]                           # 拿到一个 input
        completion = example[1]                        # 拿到一个 output

        pretokens = [i + self.prefix_cursor for i in range(0, self.prefix_len)] # 前缀字符
        intokens = [i + self.infix_cursor for i in range(0, self.infix_len)]    # 中缀字符

        conditions = pretokens + context + intokens                             # 在 input 前面加前缀字符，后面加中缀字符，本次使用前缀和中缀字符均为空 
        _input, _input_len = padding_tokens(conditions + completion, self.max_seq_length, 0, 1)    # 通过 0 字符将 conditions + completion 补足到 max_seq_length 固定长度，这是 input+output+补长字符0，_input 为模型输入，_input_len 为有效长度

        pad_targets = [0 for i in range(0, self.prefix_len)] + context + [0 for i in range(0, self.infix_len)] + completion
        _target, _ = padding_tokens(pad_targets[1:], self.max_seq_length, 0, 1) # _target 模型输出，_input 模型输入，与 _input 相比，唯一不同是向前错一位，模型输入输出均为 input+output+补长字符0，input 和 output 不是模型输入输出，是数据集中的概念

        if not self.joint_lm:                                                                      # _msk 用于匹配模型输出 _target 中哪部分未来需要计算 loss，训练时有意义
            _msk = [0.0] * (len(conditions) - 1) + [1.0] * (_input_len - len(conditions))          # 如果使用 'clm' 模式，则仅仅选取模型输出中的 output 部分计算 loss，补长字符不分不计算 loss，训练时有意义
        else:
            _msk = [1.0] * (_input_len - 1)                                                        # 如果使用 'jlm' 模式，则选取模型输出中的 input+output 部分计算 loss，补长字符不分不计算 loss，训练时有意义
        _msk, _ = padding_tokens(_msk, self.max_seq_length, 0.0, 1)
        
        output = {}
        output["id"] = torch.tensor(item, dtype=torch.long)                                        # ！！！单个整形张量，表示在数据集中的索引号
        
        _query, _query_len = padding_tokens(                                    # 通过 0 字符将 conditions 补足到 max_seq_length 固定长度，这是 input+补长字符0，_query 为模型输入，_query_len 为有效长度
            conditions, self.max_seq_length, 0, -1,                             # 当输入数据超长时，从后往前截取数据
            max_context_length = self.max_seq_length - self.max_eval_length     # 这个值用于在推理过程中限制 input 长度，为 output 留下足够空间，推理过程中 output 是一个一个字符迭代输出
        )
        output["query"] = torch.tensor(_query, dtype=torch.long)                # ！！！推理测试用的输入数据，内容为 input+补长字符0，长度为 max_seq_length，实际有效长度是 query_len
        output["query_len"] = torch.tensor(_query_len, dtype=torch.long)        # ！！！query 的实际有效长度

        output["input"] = torch.tensor(_input, dtype=torch.long)                # ！！！模型训练用的输入数据，内容为 input+output+补长字符0，长度为 max_seq_length，实际有效长度是 mask 中 1 的终止位所代表的长度
        output["target"] = torch.tensor(_target, dtype=torch.long)              # ！！！模型训练用的输出数据，内容为 input+output+补长字符0，向前错一位，长度为 max_seq_length，实际有效长度是 mask 中 1 的终止位所代表的长度

        output["mask"] = torch.tensor(_msk, dtype=torch.float)                  # ！！！_msk 用于匹配模型输出 _target 中哪部分未来需要计算 loss，训练时有意义，长度为 max_seq_length
        return output

    # 读取 jsonl 文件
    def read_ft_file(self, ft_file):                                            # ft_file 已经完成切词和映射为整数索引值
        ft_samples = []
        with open(ft_file, 'r') as reader:
            for line in reader:
                items = json.loads(line.strip())
                context = items['context']                                      # 数据集中的 input
                completion = items['completion']                                # 数据集中的 output
                ft_samples.append([context, completion])                        # [[input,output], ...]
        return ft_samples