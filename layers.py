#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List


# LoRA 基础类，只是定义了所需参数
class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False                   # 描述当前参数状态是合并还是拆分状态，所谓参数拆分是指将总 W 拆分为 Pretrain_W 和 LoRA_W
        self.merge_weights = merge_weights    # 为 True 时，训练时分开参数，测试时合并参数；为 False 时，训练时和测试时均分开参数；决定是训练还是测试由 train 和 eval 函数决定


# 该类为 LoRA 的核心类，forward 函数输入是某一个注意力模块中的 x，输出是对应 q k v 组成的矩阵。类中定义了原注意力参数 W 和对应的 LoRA 权重 A 和 B。当 merge_weights=True 时，可以通过 train() 函数实现总体参数的合并和拆分，当其等于 True 时，参数默认被拆分。
# 原注意力模块参数 W 在定义后被冻结，不能被训练，train() 函数不影响它是否可训练，该类不修改 W。W 定义在 nn.Linear 中，注意 W 的尺寸为 (in_features, out_features)，其中 in_features=n_heads*size_embding，out_features=3*n_heads*size_embding。
# 这意味着原注意模块中不同 head 的计算是不独立的，这与原 Transformer 不太一样，原来是先不同 head 独立计算，完成计算后再通过全连接层合并多头特征，而这里直接第一步就是非独立计算，这样会极大地增大参数量和计算量，它们随 head 数量二次幂增加，原本是线性增加。
# LoRA 的权重参数 A 和 B 的定义和计算过程也比较特殊，A 和 B 的合并计算是通过分组 conv1D 计算的，仔细分析计算过程，发现其实这个等效于矩阵乘法，其中分组的目的是让 Q K V 的计算相互独立，但不同 head 的计算是不独立的。A 与 B 的乘法比较特殊是一种交叉乘法，
# 相比原来的不同 head 独立计算，这二次幂级增加了计算量和输出矩阵尺寸(与二次幂增加参数量后的 W 相匹配)，但却不增加 A 和 B 的参数量。所谓交叉计算指的是不同 head 的 A 和 B 的计算不独立，会存在 head_i_A * head_j_B 的情况，这也是计算量和输出参数量增加的原因。
class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int,                     # 原注意力模块 Q/K/V 矩阵的行/列尺寸(行列尺寸相等，Q K V 尺寸相等) * n_heads，              [head1.., head2.., head3..]
        out_features: int,                    # 原注意力模块 Q K V 的输出尺寸之和(等于 3 倍的独自尺寸) * n_heads，先分 Q K V，再分 heads  [Q_head1.., Q_head2.., Q_head3.., K_head1.., K_head2.., K_head3.., V_head1.., V_head2.., Q_head3..]
        r: int = 0,                           # LoRA 的秩                                           # 从上面 in_features 和 out_features 的展示形式很容易判断，输入不区分 Q K V 但输出区分 Q K V，输入和输出都要区分不同 head
        lora_alpha: int = 1,                  # LoRA 的 alpha                                       # 所以从 in_features 映射到 out_features 的矩阵 W 是 Q K V 独立，但不同 head 不独立的
        lora_dropout: float = 0.,             # LoRA 的 Dropout，在 LoRA 分支上是，先 Dropout 再 A 和 B
        enable_lora: List[bool] = [False],    # 长度为 3，决定 Q K V 是否使用 LoRA，一般定义为 [True, False, True]，即 K 不使用 LoRA
        fan_in_fan_out: bool = False,         # 决定是否部分矩阵转置一下，设置为 True
        merge_weights: bool = True,           # 为 True 时，训练时分开参数，测试时合并参数；为 False 时，训练时和测试时均分开参数；决定是训练还是测试由 train 和 eval 模式决定
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)                                                  # 调用 nn.Linear 的基函数，实现对原有参数的定义，注意 in_features=feature*n_heads out_features=3*feature*n_heads
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)   # 调用 LoRA 基函数，实现相关参数初始化输入
        assert out_features % len(enable_lora) == 0, 'The length of enable_lora must divide out_features'              # out_features=3*feature*n_heads，Q K V 放在一起以及所有 head 放在一起了，len(enable_lora)=3 所以必须要能整除
        self.enable_lora = enable_lora                                                                                 # 长度为 3，决定 Q K V 是否使用 LoRA，一般定义为 [True, False, True]，即 K 不使用 LoRA
        self.fan_in_fan_out = fan_in_fan_out                                                                           # 决定是否部分矩阵转置一下，设置为 True
        # Actual trainable parameters                                                                  # ！！！【状态 1】 ！！！           # ！！！【状态 2】 ！！！
        if r > 0 and any(enable_lora):                                                                 # enable_lora 包含 3 True         # enable_lora 包含 2 True
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features))                             # (3*r, in_features)              # (2*r, in_features)          对于 LoRA A 是 Q K V 独自享有自己的部分，计算时相互独立
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))        # (out_features, r)               # (2/3*out_features, r)       对于 LoRA B 是 Q K V 独自享有自己的部分，计算时相互独立
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)                                                               # (3, out_features/3)             # (3, out_features/3)         全 0 矩阵
            self.lora_ind[enable_lora, :] = True                                                       # 所有元素全 True                  # 对应行所有元素全 True
            self.lora_ind = self.lora_ind.view(-1)                                                     # (out_features)                  # (out_features)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)                                        # (in_features, out_features)     # (in_features, out_features)

    # 参数初始化，尤其注意所有 LoRA 的初始化为全 0
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    # 当 Q K V 均添加 LoRA 时，该函数无意义，否则，该函数负责 padding 增加没有 LoRA 的部分
    def zero_pad(self, x):                                                                             # (out_features, in_features)     # (2/3*out_features, in_features)
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))                                       # (out_features, in_features)     # (out_features, in_features)         
        result[self.lora_ind] = x                                                                      # (out_features, in_features)     # (out_features, in_features)  对应部分有值，其他部分为 0
        return result
    
    # LoRA 权重参数 A 和 B 的合并是通过分组 conv1D 计算的，仔细分析计算过程，发现其实这个等效于矩阵乘法，其中分组的目的是让 Q K V 的计算相互独立，但不同 head 的计算不独立。与原本不同 head 独立计算相比，A 和 B 的参数量并没有增加，
    def merge_AB(self):               # 但 A*B 的计算量和输出矩阵尺寸却二次幂级增加了，原因是 A 与 B 的乘法比较特殊是一种交叉乘法，会存在 A_head_i * B_head_j 的情况，并非仅仅包含 A_head_i * B_head_i 和 A_head_j * B_head_j
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        delta_w = F.conv1d(                 # F.conv1d                                                 # enable_lora 包含 3 True          # enable_lora 包含 2 True
            self.lora_A.unsqueeze(0),       # input： (Batch_Size, In_Channel, Length)                 # (1, 3*r, in_features)            # (1, 2*r, in_features)
            self.lora_B.unsqueeze(-1),      # weight：(Out_Channel, In_Channel/Group, Kernel_Size)     # (out_features, r, 1)             # (2/3*out_features, r, 1)
            groups=sum(self.enable_lora)    # group： Group                                            # 3                                # 2  
        ).squeeze(0)                        # output：(1, Batch_Size, Out_Channel, In_Channel)         # (1, out_features, in_features)   # (1, 2/3*out_features, in_features)  无 squeeze()
        return T(self.zero_pad(delta_w))                                                               # (in_features, out_features)      # (in_features, out_features)

    # 简单点总结，这个函数的作用是：merge_weights 为 False 时不做任何处理，此时参数一定是拆分状态(前提是 merge_weights 在完成定义后不再被修改)，因为整个代码中，只有这个函数会合并参数，如果这个函数没有去合并参数，那参数一定处于拆分状态
    # merge_weights 为 True 时，如果 train mode 为 True，则使得参数处于拆分状态，如果 mode 为 False，则使得参数处于合并状态，所谓参数拆分是指将总 W 拆分为 Pretrain_W 和 LoRA_W
    def train(self, mode: bool = True):  # mode 决定是训练模式还是测试默认，默认是 True，一般都是使用默认模式，但当设置 false 时，则等效于调用 eval() 函数，此时一般直接调用 eval() 函数
        # 没用到的函数
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        # 让原有参数变为 train 模式，我理解这里其实没有造成任何影响，train 只对 BN、 Dropout 和 LoRA 产生影响，对 LoRA 产生影响时还需要结合 merge_weights 状态的设置，设置为 True 时才产生影响
        nn.Linear.train(self, mode)
        # 如果是训练模式，这里的模式不影响参数是否可训练，只对 BN、 Dropout 和 LoRA 产生影响，对 LoRA 产生影响时还需要结合 merge_weights 状态的设置，设置为 True 时才产生影响
        if mode:
            # 如果 merge_weights 为 Ture (意味着训练拆分|测试合并)，当前是训练模式，所以如果当前是合并状态则需要拆分参数，如果当前是拆分状态不需要做任何处理；如果 merge_weights 为 False，则无需做任何处理，因为默认就是拆分状态
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling                                 # (in_features, out_features)      # (in_features, out_features)
                self.merged = False
        # 如果是测试模式，这里的模式不影响参数是否可训练，只对 BN、 Dropout 和 LoRA 产生影响，对 LoRA 产生影响时还需要结合 merge_weights 状态的设置，设置为 True 时才产生影响
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling                                 # (in_features, out_features)      # (in_features, out_features)
                self.merged = True        

    # 执行函数
    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        # self.merged 描述的是参数当前合并状态
        if self.merged:  # 如果参数已经合并，那直接单次计算即可完成输出
            return F.linear(x, T(self.weight), bias=self.bias)                                         # (batch, seq_length, out_features) = (batch, seq_length, in_features) * (out_features, in_features)^T
        else:            # 如果参数还没有合并，那需要分别经过 W 和 delta_W 的计算，然后合并结果
            result = F.linear(x, T(self.weight), bias=self.bias)                                       # (batch, seq_length, out_features) = (batch, seq_length, in_features) * (out_features, in_features)^T
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling                   # (batch, seq_length, out_features) = (batch, seq_length, out_features) + (batch, seq_length, in_features) * (in_features, out_features)
            return result