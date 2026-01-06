# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...extras.misc import numpify


if TYPE_CHECKING:
    from transformers import EvalPrediction

"""
这段代码定义的 ComputeAccuracy 类是 奖励模型(Reward Model, RM) 或 偏好对齐(如 DPO, PPO) 训练中至关重要的评估模块.
在偏好学习中, "准确率"的定义与传统的分类任务不同, 它衡量的是模型给出的奖励分数是否符合人类的偏好顺序.
"""

@dataclass
class ComputeAccuracy:
    r"""
    Compute reward accuracy and support `batch_eval_metrics`.

    计算奖励模型的准确率, 并支持分批次评估指标(batch_eval_metrics).
    [为什么要这么写]: 在奖励模型训练中, 准确率定义为: Chosen 样本的分数大于 Rejected 样本分数的比例.
    """

    def _dump(self) -> dict[str, float] | None:
        """
        内部清理与汇总函数.
        [解决的问题]: 解决分布式评估时内存占用过高的问题.
        在评估大型数据集时, 我们通常分 batch 计算. 这个函数负责将当前累积的所有 batch 结果
        进行平均化处理, 并重置内部状态, 以便下一轮评估使用.
        """
        result = None
        # 如果已经存在分数值, 说明完成了一个阶段的评估, 计算均值
        if hasattr(self, "score_dict"):
            # 将列表中的布尔值转换为 float 并计算均值, 得到该阶段的平均准确率
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        # 状态重置: 初始化/清空存储字典, 确保评估结果不会在不同 epoch 间累加导致错误
        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        """
        数据类初始化后的钩子.
        [为什么要这么写]: 确保对象一经创建, 内部的 score_dict 就已就绪, 避免 AttributeError.
        """
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> dict[str, float] | None:
        """
        核心评估逻辑, 兼容 Hugging Face Trainer 的 compute_metrics 接口.

        [解决的问题]: 处理奖励模型输出的成对分数值.
        奖励模型会同时输出 chosen 和 rejected 两个序列的分数.
        """

        # 1. 硬件无关处理 (Numpify)
        # [为什么要这么写]: eval_preds.predictions 通常是 GPU 上的 PyTorch Tensor.
        # 调用 numpify 将其转换为 CPU 上的 NumPy 数组.
        # [解决的问题]: 解耦深度学习框架, 同时避免在计算指标时占用宝贵的显存, 提高评估效率.
        chosen_scores, rejected_scores = numpify(eval_preds.predictions[0]), numpify(eval_preds.predictions[1])

        # 2. 形状健壮性处理 (Shape Agnostic)
        # [为什么要这么写]: 兼容标量输出(零维阵列)和向量输出(多样本 Batch).
        # 如果是单一样本(没有 shape), 直接对比；如果是 batch, 则遍历对比.
        if not chosen_scores.shape:
            # 偏好学习的核心逻辑: 如果优选样本分数更高, 则计为"正确"(True/1)
            self.score_dict["accuracy"].append(chosen_scores > rejected_scores)
        else:
            # 遍历 Batch 中的每一对样本进行偏好对齐校验
            for i in range(len(chosen_scores)):
                self.score_dict["accuracy"].append(chosen_scores[i] > rejected_scores[i])

        # 3. 延迟计算机制 (Deferred Computation)
        # [为什么要这么写]: compute_result 参数允许用户决定是"仅累加数据"还是"立即计算结果".
        # [解决的问题]: 适配分布式训练中的汇总逻辑. 在多卡同步评估时, 通常先在各卡累加局部结果,
        # 只有在最后一个 batch 结束后才触发最终的均值计算, 减少不必要的重复运算.
        if compute_result:
            return self._dump()

"""
资深研究员视角下的技术要点总结:

准确率的本质变换:
在 LLM 偏好对齐中, 我们不关心模型打分的绝对值(例如是 10 分还是 5 分), 我们关心的是差值. 如果模型给人类喜欢的回复打了 -1 分, 给不喜欢的回复打了 -5 分, 虽然都是负数, 但模型依然是"准确"的(-1 > -5). 这段代码通过 chosen_scores > rejected_scores 完美捕捉了这一物理含义.

工程上的内存安全性:
使用 _dump 模式而不是直接返回结果, 体现了对大规模评测集的考虑. 当评估集有数万条数据时, 如果一次性将所有预测结果存入内存再计算, 会导致 OOM(内存溢出). 这种增量添加 + 最终汇总的模式是处理工业级数据的标准做法.

Hugging Face 生态适配:
eval_preds: "EvalPrediction" 的设计是为了无缝嵌入到 transformers.Trainer 中. 通过这种封装, LLaMA-Factory 能够利用原生 Trainer 的评估循环, 同时注入针对奖励模型特有的指标计算逻辑.

希望这些注释能帮助你透彻理解 LLaMA-Factory 在算法监控方面的严谨设计!
"""
