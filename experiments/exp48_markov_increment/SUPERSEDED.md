# ⚠️ 本目录结论已被 exp48.1 推翻（2026-10-27）

本目录的 `results.json` / `report.md` **原样保留**（不修改、不删除），但其核心正向结论**不成立**。

## 为什么

`exp48_markov_increment.py` 中 B2（视频边际 LOO）的实现有跨轴广播错误：

```python
loo = (cnts[None, :] * 0 + cnts)[None, :].repeat(n, axis=0) - np.eye(K)[st]
#     shape (1,1,K)                          -> (n,1,K)        广播成 (n,n,K)
q_loo = (loo + 1.0) / (n - 1 + K)                                # (n,n,K)
np.log(q_loo[np.arange(n), st]).mean()                           # (n,K) -> 平均 n*K 个元素
```

结果是把该视频**从未出现的状态**的伪概率也平均进去，B2 被系统性压低 0.33–0.65 nats。
exp48 报告的「M1−B2 = +0.869 nats/桶」主要就是这个 bug 的产物。

## 修正后的真实结果（K=4）

| 对比 | exp48 | exp48.1 |
|---|---|---|
| M1 − B2（离线视频边际） | +0.869 [0.758, 0.984] | **−0.078 [−0.111, −0.043]** |
| M1 − B2b（因果前缀，新增） | 未做 | +0.010 [−0.021, +0.042] |
| M2 − M1 | +0.059 | +0.064 [0.049, 0.080] |
| M1 − B1 | +0.366 | +0.366（不变） |

## 去哪看

- 修正实验：`experiments/exp48_1_scoring_aligned.py` + `experiments/exp48_1_scoring_aligned/report.md`
- 根因验证：`experiments/exp48_1_scoring_aligned/verify_b2_broadcast.py`

**引用纪律：任何论文/文档/汇报不得再引用 exp48 的 0.87 nats、98.6% 正向占比、以及"高弹幕组增量最大"。**
