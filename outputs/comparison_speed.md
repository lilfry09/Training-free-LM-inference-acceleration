# Speed Comparison

| dataset | split | method | context_len | ttft_sec | ttft_sec_std | tpot_sec | tpot_sec_std | e2e_sec | e2e_sec_std | throughput_tok_per_sec | throughput_tok_per_sec_std | generated_tokens | speed_repeats | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pg19 | test | baseline | 128 | 0.082925 | 0.003414 | 0.012354 | 0.000020 | 0.861248 | 0.004092 | 74.311894 | 0.352903 | 64 | 3 | metrics_baseline_pg19_test.json |
| pg19 | test | baseline | 512 | 0.235807 | 0.017307 | 0.013029 | 0.000603 | 1.056648 | 0.045447 | 60.644775 | 2.646282 | 64 | 3 | metrics_baseline_pg19_test.json |
| pg19 | test | baseline | 1024 | 0.421688 | 0.004956 | 0.014775 | 0.000178 | 1.352508 | 0.012450 | 47.322151 | 0.433397 | 64 | 3 | metrics_baseline_pg19_test.json |
| pg19 | test | gqa_reduced_kv2 | 128 | 0.095300 | 0.000847 | 0.013047 | 0.000130 | 0.917266 | 0.008349 | 69.776435 | 0.632718 | 64 | 3 | metrics_gqa_pg19_test.json |
| pg19 | test | gqa_reduced_kv2 | 512 | 0.271360 | 0.012551 | 0.014071 | 0.000054 | 1.157856 | 0.015931 | 55.281545 | 0.759914 | 64 | 3 | metrics_gqa_pg19_test.json |
| pg19 | test | gqa_reduced_kv2 | 1024 | 0.644179 | 0.040419 | 0.014264 | 0.000333 | 1.542811 | 0.059545 | 41.523439 | 1.582346 | 64 | 3 | metrics_gqa_pg19_test.json |
| pg19 | test | kvpress_knorm_r0.5 | 128 | 0.127634 | 0.026467 | 0.013632 | 0.000741 | 0.986469 | 0.066699 | 65.073239 | 4.337145 | 64 | 3 | metrics_kvpress_knorm_r0.5_pg19_test.json |
| pg19 | test | kvpress_knorm_r0.5 | 512 | 0.319038 | 0.023100 | 0.014704 | 0.001663 | 1.245416 | 0.127772 | 51.733489 | 5.046518 | 64 | 3 | metrics_kvpress_knorm_r0.5_pg19_test.json |
| pg19 | test | kvpress_knorm_r0.5 | 1024 | 0.619141 | 0.025672 | 0.015150 | 0.000440 | 1.573619 | 0.043371 | 40.691159 | 1.120762 | 64 | 3 | metrics_kvpress_knorm_r0.5_pg19_test.json |
| pg19 | test | kvpress_streamingllm_r0.5_sink4 | 128 | 0.107203 | 0.005270 | 0.013619 | 0.000730 | 0.965213 | 0.046865 | 66.408657 | 3.153667 | 64 | 3 | metrics_kvpress_streamingllm_r0.5_pg19_test.json |
| pg19 | test | kvpress_streamingllm_r0.5_sink4 | 512 | 0.279409 | 0.037219 | 0.012880 | 0.000901 | 1.090861 | 0.076386 | 58.854160 | 3.961108 | 64 | 3 | metrics_kvpress_streamingllm_r0.5_pg19_test.json |
| pg19 | test | kvpress_streamingllm_r0.5_sink4 | 1024 | 0.424589 | 0.014508 | 0.013013 | 0.000165 | 1.244398 | 0.019824 | 51.439103 | 0.812591 | 64 | 3 | metrics_kvpress_streamingllm_r0.5_pg19_test.json |
| wikitext | test | baseline | 128 | 0.118843 | 0.083761 | 0.019425 | 0.005767 | 1.342602 | 0.443368 | 50.871728 | 14.642830 | 64 | 3 | metrics_baseline_wikitext_test.json |
| wikitext | test | baseline | 512 | 0.568919 | 0.090142 | 0.020438 | 0.005078 | 1.856502 | 0.265616 | 34.950060 | 5.014091 | 64 | 3 | metrics_baseline_wikitext_test.json |
| wikitext | test | baseline | 1024 | 0.324587 | 0.037571 | 0.014732 | 0.000723 | 1.252690 | 0.080932 | 51.237877 | 3.432263 | 64 | 3 | metrics_baseline_wikitext_test.json |
| wikitext | test | gqa_reduced_kv2 | 128 | 0.233990 | 0.015087 | 0.025143 | 0.001797 | 1.818003 | 0.098693 | 35.271166 | 1.871186 | 64 | 3 | metrics_gqa_wikitext_test.json |
| wikitext | test | gqa_reduced_kv2 | 512 | 0.347096 | 0.135209 | 0.020008 | 0.007258 | 1.607582 | 0.591103 | 43.015073 | 13.075527 | 64 | 3 | metrics_gqa_wikitext_test.json |
| wikitext | test | gqa_reduced_kv2 | 1024 | 1.163051 | 0.580069 | 0.018757 | 0.002030 | 2.344731 | 0.699647 | 29.384578 | 10.515968 | 64 | 3 | metrics_gqa_wikitext_test.json |
| wikitext | test | kvpress_knorm_r0.5 | 128 | 0.116102 | 0.013648 | 0.013368 | 0.000215 | 0.958291 | 0.026827 | 66.819931 | 1.840942 | 64 | 3 | metrics_kvpress_knorm_r0.5_wikitext_test.json |
| wikitext | test | kvpress_knorm_r0.5 | 512 | 0.333139 | 0.011058 | 0.014129 | 0.000617 | 1.223286 | 0.028170 | 52.336439 | 1.195861 | 64 | 3 | metrics_kvpress_knorm_r0.5_wikitext_test.json |
| wikitext | test | kvpress_knorm_r0.5 | 1024 | 0.522897 | 0.054612 | 0.014896 | 0.002152 | 1.461330 | 0.083783 | 43.888860 | 2.437195 | 64 | 3 | metrics_kvpress_knorm_r0.5_wikitext_test.json |
| wikitext | test | kvpress_streamingllm_r0.5_sink4 | 128 | 0.118794 | 0.018960 | 0.012885 | 0.000212 | 0.930558 | 0.032236 | 68.830224 | 2.351665 | 64 | 3 | metrics_kvpress_streamingllm_r0.5_wikitext_test.json |
| wikitext | test | kvpress_streamingllm_r0.5_sink4 | 512 | 0.306238 | 0.013704 | 0.014591 | 0.001951 | 1.225458 | 0.119733 | 52.544604 | 4.903195 | 64 | 3 | metrics_kvpress_streamingllm_r0.5_wikitext_test.json |
| wikitext | test | kvpress_streamingllm_r0.5_sink4 | 1024 | 0.614020 | 0.031587 | 0.014315 | 0.000898 | 1.515866 | 0.065658 | 42.274168 | 1.872426 | 64 | 3 | metrics_kvpress_streamingllm_r0.5_wikitext_test.json |
