# Speed Comparison

| dataset | split | method | context_len | ttft_sec | tpot_sec | throughput_tok_per_sec | generated_tokens | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pg19 | test | baseline | 128 | 0.047974 | 0.015848 | 61.161793 | 64 | metrics_baseline_pg19_test.json |
| pg19 | test | baseline | 512 | 0.122562 | 0.015506 | 58.211077 | 64 | metrics_baseline_pg19_test.json |
| pg19 | test | baseline | 1024 | 0.183829 | 0.016197 | 53.144356 | 64 | metrics_baseline_pg19_test.json |
| pg19 | test | gqa_kv2 | 128 | 0.092027 | 0.038242 | 25.587273 | 64 | metrics_gqa_pg19_test.json |
| pg19 | test | gqa_kv2 | 512 | 1.172887 | 0.112514 | 7.746996 | 64 | metrics_gqa_pg19_test.json |
| pg19 | test | gqa_kv2 | 1024 | 2.902525 | 0.126371 | 5.891067 | 64 | metrics_gqa_pg19_test.json |
| pg19 | test | kvpress_knorm_r0.5 | 128 | 0.316898 | 0.085286 | 11.247980 | 64 | metrics_kvpress_knorm_r0.5_pg19_test.json |
| pg19 | test | kvpress_knorm_r0.5 | 512 | 0.967242 | 0.126438 | 7.164587 | 64 | metrics_kvpress_knorm_r0.5_pg19_test.json |
| pg19 | test | kvpress_knorm_r0.5 | 1024 | 2.020628 | 0.118865 | 6.730389 | 64 | metrics_kvpress_knorm_r0.5_pg19_test.json |
| pg19 | test | kvpress_streamingllm_r0.5_sink4 | 128 | 0.360897 | 0.104034 | 9.255177 | 64 | metrics_kvpress_streamingllm_r0.5_pg19_test.json |
| pg19 | test | kvpress_streamingllm_r0.5_sink4 | 512 | 1.156322 | 0.095398 | 8.930549 | 64 | metrics_kvpress_streamingllm_r0.5_pg19_test.json |
| pg19 | test | kvpress_streamingllm_r0.5_sink4 | 1024 | 2.665486 | 0.099294 | 7.174067 | 64 | metrics_kvpress_streamingllm_r0.5_pg19_test.json |
| wikitext | test | baseline | 128 | 0.093220 | 0.019826 | 47.681271 | 64 | metrics_baseline_wikitext_test.json |
| wikitext | test | baseline | 512 | 0.166875 | 0.019329 | 46.221905 | 64 | metrics_baseline_wikitext_test.json |
| wikitext | test | baseline | 1024 | 0.245775 | 0.021097 | 40.638543 | 64 | metrics_baseline_wikitext_test.json |
| wikitext | test | gqa_kv2 | 128 | 0.065944 | 0.020732 | 46.645401 | 64 | metrics_gqa_wikitext_test.json |
| wikitext | test | gqa_kv2 | 512 | 0.176689 | 0.022764 | 39.730935 | 64 | metrics_gqa_wikitext_test.json |
| wikitext | test | gqa_kv2 | 1024 | 0.296684 | 0.022317 | 37.588692 | 64 | metrics_gqa_wikitext_test.json |
| wikitext | test | kvpress_knorm_r0.5 | 128 | 0.062058 | 0.019466 | 49.672329 | 64 | metrics_kvpress_knorm_r0.5_wikitext_test.json |
| wikitext | test | kvpress_knorm_r0.5 | 512 | 0.168910 | 0.018862 | 47.155096 | 64 | metrics_kvpress_knorm_r0.5_wikitext_test.json |
| wikitext | test | kvpress_knorm_r0.5 | 1024 | 0.270258 | 0.020273 | 41.358709 | 64 | metrics_kvpress_knorm_r0.5_wikitext_test.json |
| wikitext | test | kvpress_streamingllm_r0.5_sink4 | 128 | 0.050185 | 0.016024 | 60.395437 | 64 | metrics_kvpress_streamingllm_r0.5_wikitext_test.json |
| wikitext | test | kvpress_streamingllm_r0.5_sink4 | 512 | 0.122852 | 0.016103 | 56.271978 | 64 | metrics_kvpress_streamingllm_r0.5_wikitext_test.json |
| wikitext | test | kvpress_streamingllm_r0.5_sink4 | 1024 | 0.186401 | 0.016632 | 51.853541 | 64 | metrics_kvpress_streamingllm_r0.5_wikitext_test.json |
