# Training-free LM Inference Acceleration

## 目录说明

- `models/`: 本地模型目录，默认使用 `pythia-70m`，不随仓库提交
- `datasets/`: 本地数据目录，包含 `wikitext-103-raw-v1` 与 `pg19_samples`，不随仓库提交
- `src/`: 评测与方法实现
- `scripts/`: 一键运行脚本
- `outputs/`: 指标输出 JSON
- `notes/`: 论文源码与 PDF

## 快速开始

在 `finalproj` 目录下运行：

```powershell
pip install -r requirements.txt
```

```powershell
.\scripts\run_eval.ps1 -Mode baseline -Dataset wikitext -Split test
```

输出会写到 `outputs/metrics_baseline_wikitext_test.json`，包含：

- `ppl_metrics.ppl`
- `cache_ppl_metrics.ppl`
- `speed_metrics[*].ttft_sec`
- `speed_metrics[*].tpot_sec`
- `speed_metrics[*].throughput_tok_per_sec`
- `kv_cache_stats`（仅 `kvpress` 会有压缩事件）
- `peak_memory_mb`（仅 CUDA）

指标说明：

- `ppl_metrics` 是 teacher-forced exact PPL，不应用 KVPress 压缩，用于和传统语言模型评测保持一致。
- `cache_ppl_metrics` 是 cached autoregressive next-token PPL，KVPress 会先压缩 prefill 阶段的 KV cache，再用压缩后的 cache 预测后续真实 token；论文中应用这个指标讨论 KVPress 的质量影响。

## 运行模式

- `baseline`: 默认 SDPA attention
- `flash`: 若 CUDA 可用则尝试 `flash_attention_2`，否则自动回退
- `gqa`: 推理时对 K/V 做分组平均（无训练近似）
- `gqa_flash`: 组合模式（CUDA+flash 才会生效）
- `kvpress`: KV 压缩模式（当前适配 `knorm` 与 `streamingllm`）

`kvpress` 示例：

```powershell
.\scripts\run_eval.ps1 -Mode kvpress -Dataset pg19 -Split test -KvpressMethod knorm -KvpressCompressionRatio 0.5
.\scripts\run_eval.ps1 -Mode kvpress -Dataset pg19 -Split test -KvpressMethod streamingllm -KvpressCompressionRatio 0.5 -KvpressNSink 4
```

## 批量实验

```powershell
.\scripts\run_matrix.ps1
```

默认会跑：

- 方法：`baseline`, `gqa`, `kvpress knorm r=0.5`, `kvpress streamingllm r=0.5 n_sink=4`
- 数据：`wikitext`, `pg19`
- split：`test`

运行结束后会生成：

- `outputs/comparison_quality.csv`
- `outputs/comparison_quality.md`
- `outputs/comparison_speed.csv`
- `outputs/comparison_speed.md`

## 复现实验

所有命令都在 `finalproj` 目录下运行。

```powershell
.\scripts\run_matrix.ps1
```

单独复现实验：

```powershell
.\scripts\run_eval.ps1 -Mode baseline -Dataset wikitext -Split test
.\scripts\run_eval.ps1 -Mode gqa -Dataset wikitext -Split test -GqaKvHeads 2
.\scripts\run_eval.ps1 -Mode kvpress -Dataset wikitext -Split test -KvpressMethod knorm -KvpressCompressionRatio 0.5
.\scripts\run_eval.ps1 -Mode kvpress -Dataset wikitext -Split test -KvpressMethod streamingllm -KvpressCompressionRatio 0.5 -KvpressNSink 4
```

默认复现设置：

- 模型：`models/pythia-70m`
- 评测 token：`MaxEvalTokens=4096`
- exact PPL 上下文：`PplContextLen=512`
- cached PPL：`CachedPplContextLen=512`, `CachedPplEvalTokens=256`
- 速度上下文：`SpeedContextLens=128,512,1024`
- 生成长度：`GenNewTokens=64`
- 当前实验环境以 CPU 为主；若无 CUDA，`flash`/`gqa_flash` 会回退到 SDPA，不作为主实验结果。

## 简短结果报告

完整结果见：

- `outputs/comparison_quality.md`
- `outputs/comparison_speed.md`

主要结论：

- `GQA` 是负结果：无训练地把 Pythia-70M 的 K/V heads 做分组平均会显著损害质量。WikiText exact PPL 从 `63.72` 上升到 `2258.01`，PG-19 exact PPL 从 `33.57` 上升到 `816.69`。
- `KVPress-Knorm` 在 WikiText 上只有轻微速度收益，throughput 提升约 `1.8%` 到 `4.2%`。
- `KVPress-StreamingLLM` 在 WikiText 上效果最好，throughput 在 context length `128/512/1024` 下分别提升约 `26.7% / 21.7% / 27.6%`。
- 在 PG-19 CPU 单样本测试中，GQA 和 KVPress 都比 baseline 慢，说明小模型 CPU 环境下 Python hook、tensor gather 和 cache 操作开销可能超过 attention 节省。
- 因此论文结论采用保守表述：KV cache compression 在部分设置下可复现地提升生成速度，但收益依赖数据集、运行环境和实现开销；GQA 在未训练适配下不适合作为成功加速方法。
