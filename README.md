# Training-free LM Inference Acceleration

## 目录说明

- `models/`: 本地模型目录，默认使用 `pythia-70m`
- `datasets/`: 本地数据目录，包含 `wikitext-103-raw-v1` 与 `pg19_samples`
- `src/`: 评测与方法实现
- `scripts/`: 一键运行脚本
- `outputs/`: 指标输出 JSON
- `notes/`: 论文源码与 PDF

## 快速开始

在 `finalproj` 目录下运行：

```powershell
pip install -r requirements.txt
```

准备模型与数据：

```powershell
.\scripts\prepare_assets.ps1
```

该命令会下载 `EleutherAI/pythia-70m` 到 `models/pythia-70m`，下载 `wikitext-103-raw-v1` 到 `datasets/wikitext-103-raw-v1`，并从 Hugging Face `pg19` 中导出每个 split 的第一个样本到 `datasets/pg19_samples`。这些本地资产体积较大，不会提交到 Git。

等价的手动准备方式：

```powershell
python .\src\prepare_assets.py --project_root .
```

如果已经有本地模型或某个数据集，可使用 `--skip_model`、`--skip_wikitext`、`--skip_pg19` 跳过对应下载。

```powershell
.\scripts\run_eval.ps1 -Mode baseline -Dataset wikitext -Split test
```

输出会写到 `outputs/metrics_baseline_wikitext_test.json`，包含：

- `ppl_metrics.ppl`
- `cache_ppl_metrics.ppl`
- `speed_metrics[*].ttft_sec`
- `speed_metrics[*].ttft_sec_std`
- `speed_metrics[*].tpot_sec`
- `speed_metrics[*].tpot_sec_std`
- `speed_metrics[*].e2e_sec`
- `speed_metrics[*].e2e_sec_std`
- `speed_metrics[*].throughput_tok_per_sec`
- `speed_metrics[*].throughput_tok_per_sec_std`
- `speed_metrics[*].raw_runs`
- `kv_cache_stats`（仅 `kvpress` 会有压缩事件）
- `gqa_cache_stats`（仅 `gqa` 会显示 reduced-KV cache shape）
- `peak_memory_mb`（仅 CUDA）

指标说明：

- `ppl_metrics` 是 chunked teacher-forced PPL：在前 `4096` tokens 上每 `512` tokens 重置一次上下文，不应用 KVPress 压缩，用于比较 baseline/GQA。
- `cache_ppl_metrics` 是 cached autoregressive next-token PPL，KVPress 会先压缩 prefill 阶段的 KV cache，再用压缩后的 cache 预测后续真实 token；论文中应用这个指标讨论 KVPress 的质量影响。
- 速度指标使用手动 greedy prefill/decode。`TTFT` 是从 prefill 开始到选出第一个 token 的 wall time；`TPOT` 是后续 decode 时间除以剩余生成 token 数；`throughput` 是 `64` 个生成 token 除以端到端时间。

## 运行模式

- `baseline`: 默认 SDPA attention
- `flash`: 若 CUDA 可用则尝试 `flash_attention_2`，否则自动回退
- `gqa`: 推理时把 8 个 K/V heads 分组平均为 2 个 cached KV heads，并用自定义 grouped attention 直接读取 reduced-KV cache
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
- speed：1 次 warm-up，3 次 measured runs，输出 mean/std/raw runs

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
- chunked PPL 上下文：`PplContextLen=512`
- cached PPL：`CachedPplContextLen=512`, `CachedPplEvalTokens=256`
- 速度上下文：`SpeedContextLens=128,512,1024`
- 生成长度：`GenNewTokens=64`
- 速度重复：`SpeedWarmupRuns=1`, `SpeedRepeats=3`
- 当前实验环境以 CPU 为主；若无 CUDA，`flash`/`gqa_flash` 会回退到 SDPA，不作为主实验结果。

cached PPL sanity check：

```powershell
.\scripts\run_cache_ppl_sanity.ps1 -Dataset wikitext -Split test
```

该检查在同一段 `512 + 256` token slice 上分别计算普通 teacher-forced continuation PPL 和 cached autoregressive PPL。当前 WikiText 结果为 `13.088025` vs `13.088739`，说明 `cache_ppl_metrics` 与同段 teacher-forced 评分一致；它和 full-window exact PPL 的差异主要来自评测 token 范围不同。

## 简短结果报告

完整结果见：

- `outputs/comparison_quality.md`
- `outputs/comparison_speed.md`
- `outputs/cache_ppl_sanity_wikitext_test.md`

主要结论（3-run mean）：

- `GQA` 是负结果：实现已改为真实 reduced-KV cache，`gqa_cache_stats` 显示每层 cache shape 为 `[1, 2, 512, 64]`，但无训练地把 Pythia-70M 改成 2 KV heads 会显著损害质量。WikiText chunked PPL 从 `63.72` 上升到 `2289.96`，PG-19 chunked PPL 从 `33.57` 上升到 `797.49`。
- `KVPress-Knorm` 在 WikiText context length `128/512` 下 throughput 分别提升约 `31.3% / 49.7%`，但在 `1024` 下下降约 `14.3%`。
- `KVPress-StreamingLLM` 在 WikiText context length `128/512` 下 throughput 分别提升约 `35.3% / 50.3%`，但在 `1024` 下下降约 `17.5%`；在 PG-19 `1024` 下提升约 `8.7%`。
- WikiText baseline 的 cached PPL 较低不是 cache 评分明显错误：同一 continuation slice 上普通 teacher-forced PPL 为 `13.088025`，cached PPL 为 `13.088739`。
- 结论采用保守表述：KV cache compression 在部分 CPU 设置下能提升生成速度，但收益依赖数据集、上下文长度和实现开销；reduced-KV GQA 在未训练适配下不适合作为成功加速方法。
