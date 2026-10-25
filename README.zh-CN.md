# Faster-R3

[English](README.md) | [简体中文](README.zh-CN.md)

**8 张 NVIDIA H100 80 GB 上，BAGEL-7B-MoT 的 R3 Tree-RL 单步总加速 2.08×；已检查的 14 步训练记录逐位一致，由 AI agent 完成优化。**

![Faster-R3 单步总加速 2.08×](docs/performance-views/h100-benchmark.svg)

| 实现 | 三个窗口的秒/步 ↓ | 最小配对加速比 ↑ |
|---|---:|---:|
| 上游代码 + 原奖励服务配置 | 30.73 / 24.72 / 27.91 | 1.00× |
| **Faster-R3 + 优化后的奖励服务配置** | **11.99 / 11.87 / 12.20** | **2.0831×** |

**总加速包含 GenEval 奖励服务的执行优化。** 两边都采用优化后奖励服务时，另一次配对测量得到训练代码本身 **1.8803×**。两者都是直接实测，没有把不同轮次的倍数相乘。表中基线取每个窗口两次基线中较快的一次。

同一台机器上，8 卡 FSDP2 ZeRO-3，BF16 计算、FP32 主权重与 Adam 状态；group size 16、policy group size 4，真实奖励 worker 与训练共享 GPU。没有改变精度，也没有用 FP8。一步包括 rollout、奖励和一次参数更新；生成步与反思编辑步交替。每次独立启动训练 16 步，前 10 步预热，后 6 步分成三个双步窗口。取三个配对比值（2.5636 / 2.0831 / 2.2869）中的最小值；不是完整训练耗时或达到同等质量所需时间。

## 快速开始

先准备[环境、公开 BAGEL 权重和 GenEval 奖励服务](docs/training.md)。在奖励服务环境中启动：

```bash
cd /path/to/reward-server
gunicorn -c /path/to/Faster-R3/scripts/reward/gunicorn_geneval.py "app_geneval:create_app()"
```

再在 Faster-R3 根目录、训练环境中运行：

```bash
export MODEL_PATH=/path/to/BAGEL-7B-MoT
export LOG_DIR=/path/to/output
export REWARD_URL=http://127.0.0.1:18085
bash scripts/train_faster_r3_geneval.sh
```

启动脚本默认 8 rank、8 shard，评测使用测试集前 8 条，应用基线与候选共用的 ZeRO-3 和加载适配。融合路径默认开启。已验证 kernel 的版本为 PyTorch 2.9.1、Triton 3.5.1、flash-attn 2.5.8。这个可移植启动脚本没有单独完成端到端复跑；原始测量由判卷台入口启动。[复现方式与内存要求](docs/training.md#launch)。

## 优化过程

最新完成版本是 **fusion_b**（源提交 `1367ad7`）：在原有静态 CUDA graph 上叠加奖励重叠、主机路径优化，以及文本、图像 flow、训练三组融合 kernel。

![8 卡优化过程，两种比较口径分开](docs/performance-views/h100-history.svg)

此前 **4 卡 1.4709×** 的结果保存在[历史报告](docs/history-4gpu.md)。硬件、分片与全局批次不同，不能与当前结果串成同一条曲线。

## 改了什么

- **奖励服务与重叠：** 每个 worker 限制为 4 个 OpenMP/MKL 线程，颜色裁剪在进程内加载，避免每次请求派生 4 个进程。等待奖励时先打包并执行参考/策略前向，计算 loss 前再收齐奖励与优势值。
- **主机与重复计算：** latent 标量每个 latent 只读一次；复用共享图像变换、block mask 与冻结 ViT 输出。保持原打包布局，跳过不参与 loss 的 CFG 副本行对应的专家 GEMM。
- **Rollout 调度：** 文本解码与主机记账流水重叠，复用相同编辑 CFG 图像前缀，显存足够时避免每次图捕获前无条件同步并清空缓存。
- **文本融合：** 残差加法与 RMSNorm、Q/K norm 与 RoPE、SiLU × up；复用层间不变索引，小投影并发。
- **图像 flow 融合：** 归一化与专家顺序恢复合并、K/V 直写合并缓存、残差/norm/MLP 融合，各 CFG 分支首次调用后捕获。
- **训练融合：** RMSNorm、SwiGLU、RoPE 前后向，以及按专家组织残差流，减少 gather、清零和 scatter。

![模块改动](docs/change-views/h100-module-map.svg)

融合实现保留所测 PyTorch 的归约顺序、BF16 舍入边界及非 FTZ 除法行为。发布的 `modeling/` 与 `train/` Python 源码和实测快照逐字节一致；附有[源码哈希与汇总记录](docs/validation/h100.json)。[开关和代码对应表](docs/training.md#changes-and-switches)。

## 验证

**逐位检查：** 总加速轮 `fusion_b_total2` 和同奖励配置轮 `fusion_b_r` 均通过 8 rank × 14 个确定性训练步的全部已记录比较：loss/统计量、奖励、优势值、rollout 文本/图像哈希、梯度范数、学习率，以及最终参数和 Adam 状态摘要。rank 0 最终摘要为 `7ee0e76bcc53ddd6b5b8d905bb99a861302d2bba9829699f101a43bd99f4d41a`。

**发布检查：** 已在单张 H100 上复跑可移植诊断脚本，37 项训练算子检查与 169 项 flow 检查通过。这是对已存档 8 卡结果的补充，不是重新跑了一轮 8 卡训练。[检查记录](docs/validation/publication-check.json)。

**计时：** 同机独立启动基线 A → 候选 → 基线 B。总加速轮两次基线最大差异 4.74%，训练代码轮为 3.51%；两者均超过 `max(1.02, 1 + 2 × spread)` 收益门。

![两种口径的配对计时](docs/performance-views/h100-paired-timing.svg)

**边界：** 14 步确定性检查不能代替长训验证；生产模式计时值未做数值比较。rank 0 的 wandb hook 记录为空，仅经 wandb 记录的评测指标没有被验证。计划的 200 步、3 种子长训及质量验证尚无完成结果。本版仍接近显存上限，不声称省显存。完整判卷台、权重、生成图像及 profile 原始记录不随仓库发布。[判卷规则与显存表](docs/judge.md)。

## 当前结果

本次发布最新完成的直接实测里程碑：**总加速 2.0831×，训练代码本身 1.8803×**。不声称已经达到性能上限；后续调度或数值改动需要重新做配对计时和正确性验证，完整训练质量仍待验证。

## 更多 AI4AI-for-Anything 项目

[Faster-BAGEL](https://github.com/AI4AI-for-Anything/Faster-BAGEL) · [Faster-YOCO](https://github.com/AI4AI-for-Anything/Faster-YOCO) · [Faster-JiT](https://github.com/AI4AI-for-Anything/Faster-JiT) · [Faster-RAE](https://github.com/AI4AI-for-Anything/Faster-RAE) · [Faster-LLaVA](https://github.com/AI4AI-for-Anything/Faster-LLaVA) · [Faster-DMD](https://github.com/AI4AI-for-Anything/Faster-DMD) · [Faster-SelfForcing](https://github.com/AI4AI-for-Anything/Faster-SelfForcing)

## 上游文档

基于 [sen-ye/R3](https://github.com/sen-ye/R3) 的 `761b0b0`，[Apache-2.0](LICENSE)。`upstream` 分支保留上游代码，安装、教程与引用见 [README.upstream.md](README.upstream.md)。
