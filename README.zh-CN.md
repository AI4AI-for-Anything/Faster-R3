# Faster-R3

[English](README.md) | [简体中文](README.zh-CN.md)

**AI agent 优化的 BAGEL-7B-MoT R3 Tree-RL 训练：在 8 张 NVIDIA H100 GPU 上每步加速 2.08×。**

![Faster-R3 单步总加速 2.08×](docs/performance-views/h100-benchmark.svg)

| 实现 | 每步 RL 耗时（秒）↓ | 加速比 ↑ |
|---|---:|---:|
| 基线 | 24.72 | 1.00× |
| **Faster-R3** | **11.87** | **2.08×** |

表格和主图展示决定最小配对加速比的第二个窗口，基线取该窗口两次基线运行中较快的一次。

**2.08× 包含 GenEval 奖励服务的执行优化**；双方使用相同优化后奖励配置时，另一次配对测量得到训练代码本身加速 **1.88×**。

在 **8 张 NVIDIA H100 GPU（80 GB）** 上进行短程测试，使用 FSDP2 ZeRO-3、BF16 计算、FP32 主权重与 Adam 状态，GenEval 奖励 worker 与训练共享 GPU。group size 为 16，policy group size 为 4；精度保持不变，**不使用 FP8**。

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

## Agent 优化过程

优化涵盖静态 CUDA graph、奖励重叠、主机路径，以及文本、图像 flow 和训练的融合 kernel。图中每个阶段都在同一台 8 卡机器上与新测基线配对比较。三组融合分别在同一个未融合版本上测试，再组合为最终配置。

![8 卡优化过程，两种比较口径分开](docs/performance-views/h100-history.svg)

此前 **4 卡 1.4709×** 的结果保存在[历史报告](docs/history-4gpu.md)。硬件、分片与全局批次不同，不能与当前结果串成同一条曲线。

## 主要改动

- **奖励服务与重叠：** 每个 worker 限制为 4 个 OpenMP/MKL 线程，颜色裁剪在进程内加载，避免每次请求派生 4 个进程。等待奖励时先打包并执行参考/策略前向，计算 loss 前再收齐奖励与优势值。
- **主机与重复计算：** latent 标量每个 latent 只读一次；复用共享图像变换、block mask 与冻结 ViT 输出。保持原打包布局，跳过不参与 loss 的 CFG 副本行对应的专家 GEMM。
- **Rollout 调度：** 文本解码与主机记账流水重叠，复用相同编辑 CFG 图像前缀，显存足够时避免每次图捕获前无条件同步并清空缓存。
- **文本融合：** 残差加法与 RMSNorm、Q/K norm 与 RoPE、SiLU × up；复用层间不变索引，小投影并发。
- **图像 flow 融合：** 归一化与专家顺序恢复合并、K/V 直写合并缓存、残差/norm/MLP 融合，各 CFG 分支首次调用后捕获。
- **训练融合：** RMSNorm、SwiGLU、RoPE 前后向，以及按专家组织残差流，减少 gather、清零和 scatter。

![模块改动](docs/change-views/h100-module-map.svg)

融合实现保留所测 PyTorch 的归约顺序、BF16 舍入边界及非 FTZ 除法行为。发布的 `modeling/` 与 `train/` Python 源码和实测快照逐字节一致；附有[源码哈希与汇总记录](docs/validation/h100.json)。[开关和代码对应表](docs/training.md#changes-and-switches)。

## 正确性验证

**数值检查。** 两种比较配置在 **全部 8 rank 的 14 个确定性训练步（0–13）** 中，所有已记录项目均逐位一致：训练 loss/统计量、奖励、优势值、rollout 文本/图像哈希、梯度范数、学习率，以及最终参数和 Adam 状态摘要。

**算子检查。** 单张 H100 上的补充诊断通过了 37 项训练算子检查与 169 项图像 flow 检查，作为 8 rank 训练检查的补充。详见[检查记录](docs/validation/publication-check.json)。

**计时。** 在同一台机器上依次独立启动基线 A → Faster-R3 → 基线 B。每次运行 16 步：前 10 步预热，后 6 步分成三个双步窗口，各包含一个生成步和一个反思编辑步。每步包括 rollout、奖励和一次参数更新。每个窗口取两次基线中较快的一次，再取三个配对加速比中的最小值。

| 窗口（训练步） | 基线（秒/步）↓ | Faster-R3（秒/步）↓ | 配对加速比 ↑ |
|---|---:|---:|---:|
| 1（10–11） | 30.73 | 11.99 | 2.5636× |
| **2（12–13）** | **24.72** | **11.87** | **2.0831×** |
| 3（14–15） | 27.91 | 12.20 | 2.2869× |

完整配置的最小加速比为 **2.0831×**，训练代码的独立配对比较为 **1.8803×**。加速比使用未舍入的计时值计算，表中耗时保留两位小数。

![两种口径的配对计时](docs/performance-views/h100-paired-timing.svg)

**验证范围。** 数值比较覆盖上述确定性训练步；生产模式计时值和仅经 wandb 记录的评测指标未做比较。加速比衡量短程 RL 单步耗时，完整训练耗时与达到同等质量所需时间未测量。显存占用仍接近设备上限。详见[比较记录与验收标准](docs/judge.md)和[显存表](docs/judge.md#memory)。

## 更多 AI4AI-for-Anything 项目

[Faster-BAGEL](https://github.com/AI4AI-for-Anything/Faster-BAGEL) · [Faster-YOCO](https://github.com/AI4AI-for-Anything/Faster-YOCO) · [Faster-JiT](https://github.com/AI4AI-for-Anything/Faster-JiT) · [Faster-RAE](https://github.com/AI4AI-for-Anything/Faster-RAE) · [Faster-LLaVA](https://github.com/AI4AI-for-Anything/Faster-LLaVA) · [Faster-DMD](https://github.com/AI4AI-for-Anything/Faster-DMD) · [Faster-SelfForcing](https://github.com/AI4AI-for-Anything/Faster-SelfForcing)

## 上游文档

基于 [sen-ye/R3](https://github.com/sen-ye/R3) 的 `761b0b0`，[Apache-2.0](LICENSE)。`upstream` 分支保留上游代码，安装、教程与引用见 [README.upstream.md](README.upstream.md)。
