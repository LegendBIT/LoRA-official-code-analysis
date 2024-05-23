#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  L，icensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import math
import os
from collections import OrderedDict 
import copy
import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parameter import Parameter

import loralib as lora


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


# LN 归一化层，注意这里的被归一化目标是单 batch 单 token 所有 head 的特征量，更要注意 weight 矢量和 bias 矢量与被归一化目标同尺寸，这意味着 x 最后一维每个数字都对应一个 weight 值和一个 bias 值，而不是整个最后一维就一个 weight 值和 bias 值
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))   # (nx,)
        self.bias = nn.Parameter(torch.zeros(hidden_size))    # (nx,)
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)                          # (batch, seq_length, nx)          nx=n_heads*size_embding
        s = (x - u).pow(2).mean(-1, keepdim=True)             # (batch, seq_length, nx)          nx=n_heads*size_embding
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)   # (batch, seq_length, nx)          nx=n_heads*size_embding
        return self.weight * x + self.bias                    # (batch, seq_length, nx)          nx=n_heads*size_embding


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


# 该类为注意力机制的核心类，完整实现了整个注意力机制包括两部分。第一部分是 q k v 的计算，第二部分是注意力结果计算。第一部分融合进了 LoRA 训练策略，可以实现 LoRA 训练，可以合并参数也可以拆分参数。第二部分不在乎有没有 LoRA，就是纯注意力计算的结果。
# 该类的实现与标准 transformer 不太一样，标准 transformer 在计算 q k v 时每个 head 独自计算，相互不干扰，例如 head1_q=head1_Q*head1_x，不需要 head2_x 参与，而该类在计算时不同 head 相互干扰，具体说就是 head1_q 在计算时不仅仅通过 head1_x 计算，
# 还需要 head2_x，... 等参与计算，例如 [head1_q, head2_q]=Q*[head1_x, head2_x] 这会造成 Q 参数量和计算量的二次幂级增加，原本不同 head 独立计算时，只需要 n_head 个矩阵即可完成不同 head 的 q 的计算，但该类则需要 n_head*n_head 个矩阵组成的大 Q
# 进行计算。此外，在第一步中的 LoRA 也没有遵守标准模式，因为本就不存在单独的 head1_Q, head2_Q ... 矩阵，只存在一个大的 Q 矩阵，所以 A 和 B 也是多 head 不独立的，直接用 [head1_A, head2_A] 与 [head1_B, head2_B] 计算 delta_Q, 计算量二次幂级增加。
# 训练时，一次性输入长度 n 的序列，输出也是长度 n 的序列，其中前向注意力机制通过 mask 实现，且不存在 k v 的缓存，即不存在历史过去的 k v，输入序列长度固定为最大长度，真实序列长度不足时需要补特殊字符，一个 batch 中不要求所有序列都等长，存在算力浪费。
# 测试时，首先一次性输入长度 n 的输入序列，序列长度固定为最大长度，真实序列长度不足时需要补特殊字符，一个 batch 中不要求所有序列都等长，需要有个参数 len 记录 batch 中每个样本的真实长度，第一次推理结束后，每个样本从 len 记录的最后一个字符处拿到新的
# 第一个输出字符(作为第一轮推理的输出)，然后将这个字符(长度固定为 1)再送入模型进行推理得到结果(作为第二轮推理的输出)，计算时，原已经计算过 k v 的字符不用再次计算，只需从缓存中提取即可，本次仅计算单个新字符的 q k v，计算完成后需要将新 k v 缓存以备
# 下一个新字符的查询，缓存 k v 的数据结构 past 的长度在模型针对这个样本的第一次推理时就固定为输入序列的最大长度了，后续更新缓存时，只需要在上一步真实长度 +1 处填上新的 k v 即可，模型训练时 batch=batchsize，但测试时，batch=batchsize*num_beams
class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):          # nx - embedding lens 是单个 token 特征长度乘以 n_heads 后的结果;  n_ctx - input lens  n_ctx = max_seq_length 当实际句子长度小于 max_seq_length 时需要补特殊字符
        super(Attention, self).__init__()
        n_state = nx                                             # in Attention: n_state = 768 = 64*12 (nx = n_embd)
        
        assert n_state % config.n_head == 0                      # nx = heads*in_fact_lens = 12 * 64
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))  # (1, 1, n_ctx, n_ctx)     下三角全 1 上三角全 0 的屏蔽矩阵，用于实现单向注意力机制
        self.n_head = config.n_head
        self.split_size = n_state                                # nx
        self.scale = scale
        self.c_attn = lora.MergedLinear(                         # 该类为 LoRA 的核心类，forward 函数输入是某一个注意力模块中的 x，输出是对应 q k v，类中定义了原注意力参数 W 和对应的 LoRA 权重 A 和 B
            nx, n_state*3,                                       # nx = heads*in_fact_lens， n_state*3 = 3*heads*in_fact_lens
            r=config.lora_attn_dim, 
            lora_alpha=config.lora_attn_alpha, 
            lora_dropout=config.lora_dropout, 
            enable_lora=[True, False, True], 
            fan_in_fan_out=True,
            merge_weights=False                                  # 在 merge_weights=True 时可以通过 train() 函数实现总体参数的合并和拆分，在 merge_weights=True 时，默认参数拆分
        )
        self.c_proj = Conv1D(n_state, nx)                        # (nx, nx) 相比原 Transformer，计算量和参数量指数增加了，原本是合并为一个 head 后再全连接计算，现在是不合并就全连接计算，注意力矩阵计算也是这样，所以整体计算量和参数量指数级增加

        self.config = config
    
    # q : (batch, head, q_seq_length, head_features)
    # k : (batch, head, head_features, max_seq_length)
    # w : (batch, head, q_seq_length, max_seq_length)
    # v : (batch, head, max_seq_length, head_features)
    # 解码器的 attention；训练时，q_seq_length=max_seq_length，每个 token 同时都需要参与计算；测试时，q_seq_length=1，只有最后一个 token 需要参与计算，所有 token 是迭代着输入输出；不同 head 分开独立计算
    # 在计算得到 q k v 的过程时，不同 head 是不独立的，但是在基于 q k v 计算注意力结果时，不同 head 是独立的，单独计算，且这里是 masked 注意力机制，需要屏蔽部分权重
    def _attn(self, q, k, v, len_kv=None):
        
        # 1. 计算得到权重系数矩阵 w
        w = torch.matmul(q, k)                                   # (batch, head, q_seq_length, max_seq_length) = (batch, head, q_seq_length, head_features) * (batch, head, head_features, max_seq_length)
        if self.scale:
            w = w / math.sqrt(v.size(-1))                        # (batch, head, q_seq_length, max_seq_length)
        
        # 2. 对权重系数矩阵进行 mask，以实现每个 token 只能看到它前面的 tokens 的功能，如果没有这部分，那就是编码器的 atten 了，这部分只对训练时起作用，测试时不起作用
        nd, ns = w.size(-2), w.size(-1)                          # q_seq_length, max_seq_length                  训练时 q_seq_length = max_seq_length          测试时 q_seq_length = 1
        b = self.bias[:, :, ns-nd:ns, :ns]                       # (1, 1, q_seq_length, max_seq_length)          训练时 后两维是下三角全 1 矩阵                  测试时 第三维是 qv_seq_length 第四维全 1                                               
        w = w * b - 1e10 * (1 - b)                               # (batch, head, q_seq_length, max_seq_length)   训练时 取 w 的下三角矩阵部分，上三角为 -1e10     测试时 第三维长度是 1 第四维全 1             (-1e10 的作用是经过 Softmax 后变 0)
        
        # 3. 检查是否对输入长度做出了限制，如果有限制，则把系数矩阵中超出长度的部分变为 -1e10，主要对测试时起到长度控制作用，因为测试时是一个一个 token 的输出，len_kv 是逐步递增的，训练时不起作用
        if len_kv is not None:                                                 # size=1，type=tensor， eg. torch.tensor((3,))
            _len = torch.arange(k.size(-1), device=k.device)                   # (max_seq_length)     数值 0..max_seq_length-1
            _input_msk =  _len[None, :] >= (len_kv)[:, None]                   # (1, max_seq_length)  数值前 len_kv 为 False 后面为 True
            w = w.masked_fill(_input_msk.unsqueeze(1).unsqueeze(2), -1.0e10)   # (batch, head, q_seq_length, max_seq_length)  最后一维数值前 len_kv 为原值，后面为 -1e10  (-1e10 的作用是经过 Softmax 后变 0)
        
        # 4. 得到最终的权重系数矩阵
        w = nn.Softmax(dim=-1)(w)                                              # (batch, head, q_seq_length, max_seq_length)

        # 5. 得到最终的 atten 输出
        return torch.matmul(w, v)                                              # (batch, head, q_seq_length, head_features) = (batch, head, q_seq_length, max_seq_length) * (batch, head, max_seq_length, head_features)

    # 将不同的 head 的特征合在一起
    def merge_heads(self, x):                                     # (batch, head, seq_length, head_features)
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)  # (batch, seq_length, head*head_features)
        return x.view(*new_x_shape)                               # (batch, seq_length, head*head_features)

    # 将不同的 head 的特征分开
    def split_heads(self, x, k=False):                            # (batch, seq_length, head*head_features)
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1).contiguous()             # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3).contiguous()             # (batch, head, seq_length, head_features)

    # 执行函数
    # 训练时，layer_past=None, len_past=None, q_seq_length=max_seq_length, 序列整体输入整体输出，通过 mask 实现前向注意力，不用序列一个一个输入输出
    # 测试时，layer_past=xxxx, len_past=n-1,  q_seq_length=1,  在输入第 n 个字符时，前 n-1 个字符已经完成自己的 q k v 的计算，所以无需重新计算，只需要计算第 n 个字符的 q k v 即可
    def forward(self, x, history=None, layer_past=None, len_past=None):
        hidden_states = x                                         # (batch, q_seq_length, in_features)          in_features=n_heads*size_embding

        x = self.c_attn(x)                                        # (batch, q_seq_length, out_features)         out_features=3*n_heads*size_embding        完成 q k v 的计算
        query, key, value = x.split(self.split_size, dim=2)       # (batch, q_seq_length, in_features), (batch, q_seq_length, in_features), (batch, q_seq_length, in_features) <-- (batch, q_seq_length, out_features)

        query = self.split_heads(query)                           # (batch, head, q_seq_length, head_features)  head_features=size_embding
        key = self.split_heads(key, k=True)                       # (batch, head, head_features, q_seq_length)
        value = self.split_heads(value)                           # (batch, head, q_seq_length, head_features)

        #_input_msk = None

        len_kv = None
        # 训练时，layer_past=None，测试时，layer_past=xxxx
        if layer_past is not None:
            # 当存在 layer_past 却不存在 len_past 时，只需要将当前的 key 和 value 添加在 past_key 和 past_value 后面，实现过往 key 和 value 的延长，key 和 value 的长度就是当前已输入序列的实际长度，当前代码永远不会执行这句话
            if len_past is None:
                past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]         # (batch, head, head_features, n-1), (batch, head, n-1, head_features)
                key = torch.cat((past_key, key), dim=-1)                                      # (batch, head, head_features, n)        
                value = torch.cat((past_value, value), dim=-2)                                # (batch, head, n, head_features)
            # 当存在 layer_past 和 len_past 时，past_key 和 past_value 是按照最长序列事先定义一个列表，然后只需要将当前的 key 和 value 添加在 past_key 和 past_value 中对应位置即可，layer_past 举例 [2, 10, 12, 512, 64]
            else:                        # past_key 和 past_value 是事先定义一个列表，其实就是针对样本的第一次推理时计算的 key 和 value，第一次推理时输入长度是 max_seq_length，所以 key 和 value 也是长度 max_seq_length
                key_seq = key.shape[-1]  # 针对样本的第二次推理开始，输入长度固定为 1
                assert key_seq == 1                                                           # seq_length=1

                _batch = torch.arange(0, key.shape[0], dtype=torch.long, device=key.device)   # batch

                past_key, past_value = layer_past[0], layer_past[1]                           # (batch, head, max_seq_length, head_features), (batch, head, max_seq_length, head_features)

                past_key[_batch,:,len_past,:] = key.squeeze(-1)                               # (batch, head, max_seq_length, head_features) <-- (batch, head, head_features) len_past
                past_value[_batch,:,len_past,:] = value.squeeze(-2)                           # (batch, head, max_seq_length, head_features) <-- (batch, head, head_features) len_past

                key = past_key.transpose(-2, -1)                                              # (batch, head, head_features, max_seq_length)
                value = past_value                                                            # (batch, head, max_seq_length, head_features)

                len_kv = len_past + 1                                                         # 整数

        present = torch.stack((key.transpose(-2, -1), value))                                 # (2, batch, head, max_seq_length, head_features)   max_seq_length = n_ctx      present 举例 [2, 10, 12, 512, 64]
        a = self._attn(query, key, value, len_kv = len_kv)                                    # (batch, head, q_seq_length, head_features)        q_seq_length = max_seq_length or 1
        a = self.merge_heads(a)                                                               # (batch, q_seq_length, head*head_features)
        a = self.c_proj(a)                                                                    # (batch, q_seq_length, head*head_features)         head*head_features = nx
        return a, present                                                                     # (batch, q_seq_length, head*head_features), (2, batch, head, max_seq_length, head_features)

# 两个全连接层，先升维再降维
class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2

# 整个 Transformer 的 decoder 的基础模块，LoRA 隐藏在注意力模块中的第一步，即 q k v 的计算过程中
class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)  # LN 归一化层，注意这里的被归一化目标是单 batch 单 token 所有 head 的特征量，更要注意 weight矢量 和 bias矢量与被归一化目标同尺寸
        self.attn = Attention(nx, n_ctx, config, scale)           # 该类为注意力机制的核心类，完整实现了整个注意力机制，输入是 nx 输出也是 nx，nx = head*head_features，完成注意力特征提取、特征融合、以及全连接层映射
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)  # LN 归一化层，注意这里的被归一化目标是单 batch 单 token 所有 head 的特征量，更要注意 weight矢量 和 bias矢量与被归一化目标同尺寸
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None, len_past=None):                                     # (batch, q_seq_length, nx)          nx=n_heads*size_embding
        a, present = self.attn(self.ln_1(x), layer_past=layer_past, len_past=len_past)        # (batch, q_seq_length, head*head_features), (2, batch, head, max_seq_length, head_features)
        x = x + a                                                                             # (batch, q_seq_length, head*head_features)
        m = self.mlp(self.ln_2(x))                                                            # (batch, q_seq_length, head*head_features)                                 
        x = x + m                                                                             # (batch, q_seq_length, head*head_features)
        return x, present                                                                     # (batch, q_seq_length, head*head_features), (2, batch, head, max_seq_length, head_features)


# GPT2 的 backbone
class GPT2Model(nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.n_layer = config.n_layer                                                         # 层数
        self.n_embd = config.n_embd                                                           # 头数*单 token 特征  head*head_features
        self.n_vocab = config.vocab_size                                                      # 输出/输出词汇表长度

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)                             # (vocab_size, head*head_features)  输入/输出特征嵌入矩阵，实现 token id 到特征向量的映射，类似于 word2vector
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)                            # (n_positions, head*head_features) 输入位置嵌入矩阵
        block = Block(config.n_ctx, config, scale=True)                                       # Transformer 的 decoder 的基础模块
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])         # n_layer 个 decoder 组成的列表
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)                   # LN 模块

        self.config = config

    # GPT2 的 backbone 的正向推理，模型训练时 batch=batchsize，测试时，batch=batchsize*num_beams，训练时，q_seq_length=max_seq_length，测试时，q_seq_length=1
    def forward(
        self, 
        input_ids,                              # 训练时 (batch, max_seq_length)  测试时第一次输入 (batch, max_seq_length) 后续输入 (batch, 1)  对应的是 token_id
        position_ids=None, 
        token_type_ids=None, 
        past=None,                              # past 为一个列表，长度是 n_layer，里面每个 tensor 大小是 (2, batch, head, max_seq_length, head_features)，第一维是 k v 区分，max_seq_length 最长输入序列长度，用于缓存历史 k v
        len_past=None                           # (batch,) 里面每个元素是一个整数，对应 past 的 max_seq_length，max_seq_length 是最长输入序列长度，len_past 表示实际有效长度
    ):
        if past is None:                        # 如果 past 为空，说明没有任何历史缓存 k v，所以 past_length=0
            past_length = 0                     
            past = [None] * len(self.h)         # 那每一层都需要设为空，每一层都没有任何历史缓存 k v，其中 len(self.h) = n_layer
        elif len_past is None:                  # 如果 len_past 为空，past 不为空，那说明 past 的长度不是 max_seq_length 而是实际长度，这个在当前代码下不会发生！！！
            # equal size for past. []
            past_length = past[0][0].size(-2)   # past 的实际长度

        if position_ids is None and len_past is None:                                # 说明是训练模式或者测试模式下的第一次推理
            position_ids = torch.arange(
                past_length, input_ids.size(-1) + past_length,                       # (max_seq_length,)  [0, 1, ..., max_seq_length-1]  位置 id
                dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)            # (batch, max_seq_length)
        elif len_past is not None:                                                   # 说明是测试模式下第二次推理或以后的推理
            position_ids = (len_past).unsqueeze(1) #.long()                          # (batch, 1)  

        input_shape = input_ids.size()                                               # (batch, q_seq_length)
        input_ids = input_ids.view(-1, input_ids.size(-1))                           # (batch, q_seq_length)
        position_ids = position_ids.view(-1, position_ids.size(-1))                  # (batch, q_seq_length)

        inputs_embeds = self.wte(input_ids)                                          # (batch, q_seq_length, head*head_features) <-- (batch, q_seq_length)  这就是输入特征

        position_embeds = self.wpe(position_ids)                                     # (batch, q_seq_length, head*head_features) <-- (batch, q_seq_length)  这就是位置编码

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))        # (batch, q_seq_length)
            token_type_embeds = self.wte(token_type_ids)                             # (batch, q_seq_length, head*head_features) <-- (batch, q_seq_length)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds          # (batch, q_seq_length, head*head_features)
        presents = []
        # 搭建 n_layer 层 decoder 组成的模型，len(self.h) = n_layer
        for block, layer_past in zip(self.h, past):                                  # (2, batch, head, max_seq_length, head_features)，第一维是 k v 区分，max_seq_length 最长输入序列长度
            hidden_states, present = block(hidden_states, layer_past = layer_past, len_past=len_past)  # (batch, q_seq_length, head*head_features), (2, batch, head, max_seq_length, head_features)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)                                     # (batch, q_seq_length, head*head_features)
        output_shape = input_shape + (hidden_states.size(-1),)                       # (batch, q_seq_length, head*head_features)
        return hidden_states.view(*output_shape), presents                           # (batch, q_seq_length, head*head_features), (2, batch, head, max_seq_length, head_features)


# GPT2 的 head，模型训练时 batch=batchsize，测试时，batch=batchsize*num_beams，训练时，q_seq_length=max_seq_length，测试时，q_seq_length=1
class GPT2LMHead(nn.Module):
    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):                      # (vocab_size, head*head_features)
        embed_shape = model_embeddings_weights.shape                                 # (vocab_size, head*head_features)
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights                               # Tied weights，将输入和输出特征嵌入矩阵绑定，即 (id --> feature) <==> (feature --> id)

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)                                       # (batch, q_seq_length, vocab_size)
        return lm_logits


# 配置类
class GPT2Config(object):
    def __init__(
        self,
        vocab_size_or_config_json_file=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        lora_attn_dim=0,
        lora_attn_alpha=128,
        lora_dropout=0.0,
        lora_r_dropout=0.0,
        fix_dropout=0.0,
    ):
        self.vocab_size = vocab_size_or_config_json_file
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.lora_attn_dim = lora_attn_dim
        self.lora_attn_alpha = lora_attn_alpha
        self.lora_dropout = lora_dropout
        self.lora_r_dropout = lora_r_dropout

        self.fix_dropout = fix_dropout


# GPT2 完整类
# 训练时，一次性输入长度 n 的序列，输出也是长度 n 的序列，其中前向注意力机制通过 mask 实现，且不存在 k v 的缓存，即不存在历史过去的 k v，输入序列长度固定为最大长度，真实序列长度不足时需要补特殊字符，一个 batch 中不要求所有序列都等长，存在算力浪费
# 测试时，首先一次性输入长度 n 的输入序列，序列长度固定为最大长度，真实序列长度不足时需要补特殊字符，一个 batch 中不要求所有序列都等长，需要有个参数 len 记录 batch 中每个样本的真实长度，第一次推理结束后，每个样本从 len 记录的最后一个字符处拿到新的
# 第一个输出字符(作为第一轮推理的输出)，然后将这个字符(长度固定为 1)再送入模型进行推理得到结果(作为第二轮推理的输出)，计算时，原已经计算过 k v 的字符不用再次计算，只需从缓存中提取即可，本次仅计算单个新字符的 q k v，计算完成后需要将新 k v 缓存以备
# 下一个新字符的查询，缓存 k v 的数据结构 past 的长度在模型针对这个样本的第一次推理时就固定为输入序列的最大长度了，后续更新缓存时，只需要在上一步真实长度 +1 处填上新的 k v 即可
class GPT2LMModel(nn.Module):
    def __init__(self, config):
        super(GPT2LMModel, self).__init__()
        self.transformer = GPT2Model(config)                             # GPT2 backbone
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)   # GPT2 head
        self.apply(self._init_weights)

    def set_tied(self):                                                  # 绑定输入输出位置嵌入矩阵
        """ Make sure we are sharing the embeddings"""
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    # 正向推理函数，模型训练时 batch=batchsize，测试时，batch=batchsize*num_beams，训练时，q_seq_length=max_seq_length，测试时，q_seq_length=1
    def forward(
        self, 
        input_ids,                  # 训练时 (batch, max_seq_length)  测试时第一次输入 (batch, max_seq_length) 后续输入 (batch, 1)  对应的是 token_id  训练时数据形式 input+output+补长字符0 测试时数据形式 input+补长字符0
        lm_labels=None,             # 训练时 (batch, max_seq_length)  测试时 None  其实就是 input_ids 向前错一位  训练时数据形式 input+output+补长字符0  input 和 output 是数据集中的概念，不是模型的输入输出
        lm_mask=None,               # 训练时 (batch, max_seq_length)  测试时 None  用于描述 lm_labels 中哪部分是数据集中的 input 哪部分是数据集 中的 output，仅仅计算 output 部分的 loss
        past=None,                  # 训练时 None  测试时 (2, batch, head, max_seq_length, head_features) 第一维是 k v 区分，max_seq_length 最长输入序列长度，用于缓存历史 k v
        len_past=None,              # 训练时 None  测试时 (batch,) 用于记录 past 中的实际长度
        label_smooth=0.0,           # 带有 smooth 的交叉熵损失函数的 smooth 权重
        is_report_accuracy=False    # 是否输出两个结果，batch 中每个样本的预测结果中只要有一个 token 被击中则设置为 1，batch 中每个样本的预测结果中所有 token 都被击中则设置为 1
    ):
        _batch, _len = input_ids.shape                                                        #  batch, q_seq_length
        hidden_states, presents = self.transformer(input_ids, past=past, len_past=len_past)   # (batch, q_seq_length, head*head_features), (2, batch, head, max_seq_length, head_features)
        lm_logits = self.lm_head(hidden_states)                                               # (batch, q_seq_length, vocab_size) 未经过 softmax

        if lm_labels is not None:

            if is_report_accuracy:
                _pred_token = torch.argmax(lm_logits, dim=-1)                                 # (batch, max_seq_length)
                _hit = (_pred_token == lm_labels) * lm_mask                                   # (batch, max_seq_length)  一个 batch 中多少个 token 完全预测正确，正确就是击中 hit

                _t1_acc = torch.zeros(_batch, dtype=torch.float, device=input_ids.device)     # (batch,)  batch 中每个样本的预测结果中只要有一个 token 被击中则设置为 1
                _all_acc = torch.zeros(_batch, dtype=torch.float, device=input_ids.device)    # (batch,)  batch 中每个样本的预测结果中所有 token 都被击中则设置为 1
                
                for _b in range(0, _batch):                                                   # 遍历 batch
                    for _i in range(0, _len):                                                 # 遍历 max_seq_length
                        if lm_mask[_b, _i] >= 1.0:                                            # 判断当前 token 是否是数据集中的 output 部分
                            if _hit[_b, _i] > 0:                                              # 当前 token 是否预测正确，即是否击中
                                _t1_acc[_b] = 1.0                                             # batch 中每个样本的预测结果中只要有一个 token 被击中则设置为 1
                            break  

                    _is_succ = True
                    for _i in range(0, _len):                                                 # 遍历 max_seq_length
                        if lm_mask[_b, _i] >= 1.0:                                            # 判断当前 token 是否是数据集中的 output 部分
                            if _hit[_b, _i] <= 0:                                             # 当前 token 是否预测正确，即是否击中                                                                
                                _is_succ = False                                              # batch 中每个样本的预测结果中只要有一个 token 未被击中则设置为 False
                                break

                    if _is_succ:
                        _all_acc[_b] = 1.0                                                    # batch 中每个样本的预测结果中所有 token 都被击中则设置为 1

                #_t1_acc = _t1_acc * 1.0 / _batch
                #_all_acc = _all_acc * 1.0 / _batch

            if label_smooth > 0.0001:
                logprobs = torch.nn.functional.log_softmax(lm_logits.view(-1, lm_logits.size(-1)), dim=-1)   # (batch*q_seq_length, vocab_size) 经过 log_softmax
                nll_loss = -logprobs.gather(dim=-1, index=lm_labels.view(-1).unsqueeze(1))                   # (batch*q_seq_length, 1) 经过 -log_softmax，基于 token_id 用lm_labels从logprobs最后一维中提取出对应的概率值
                nll_loss = nll_loss.squeeze(1)                                                               # (batch*q_seq_length,) 每一个值都是 vocab_size 中正确 token 所对应位置处的概率值
                smooth_loss = -logprobs.mean(dim=-1)                                                         # (batch*q_seq_length,) 经过 -log_softmax 后，最后一维的均值
                loss = (1.0 - label_smooth) * nll_loss + label_smooth * smooth_loss                          # nll_loss 是最后一维正确 token 所对应位置处的概率值，smooth_loss 是最后一维所有概率值的均值，都经过了 -log_softmax
                loss = loss.view(_batch, _len)                                                               # (batch, q_seq_length）带有 smooth 的交叉熵损失
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduce=False)
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)).view(_batch, _len) # (batch, q_seq_length）交叉熵损失

            if lm_mask is None:
                lm_mask = torch.ones(loss.shape, dtype=loss.dtype, device=loss.device)
            loss = loss * lm_mask 

            loss = loss.sum() / (lm_mask.sum() + 0.0001)                                                       # 仅仅计算 output 部分的 loss

            if is_report_accuracy:
                return lm_logits, loss, _t1_acc, _all_acc   # 训练模式
            else:
                return lm_logits, loss                      # 训练模式
        return lm_logits, presents                          # 测试模式 (batch, q_seq_length, vocab_size), (2, batch, head, max_seq_length, head_features) 返回模型推理结果和 k v 缓存结果
    
    # 初始化权重，不包含 LoRA 权重
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # 上载权重
    def load_weight(self, state_dict):
        # 原始权重跟自定义的权重在名称上有些不同，所以需要修改下名称
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
    
        state_dict_tmp = copy.deepcopy(state_dict)
        old_keys = []
        new_keys = []
        for key in state_dict_tmp:
            new_key = None
            if key.endswith(".g"):
                new_key = key[:-2] + ".weight"
            elif key.endswith(".b"):
                new_key = key[:-2] + ".bias"
            elif key.endswith(".w"):
                new_key = key[:-2] + ".weight"
            
            if key.startswith("module.transformer."):
                new_key = key[len("module.transformer."):]

            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        # 完成权重名称修改
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        # 将自定义的随机初始化的 LoRA 相关权重也写进权重中
        for n, p in self.transformer.named_parameters():
            if n not in state_dict:
                state_dict[n] = p
        # 上载权重，其实可以把 strict 设置为 False
        self.transformer.load_state_dict(state_dict, strict=False)
        self.set_tied()  # 绑定输入输出位置嵌入矩阵