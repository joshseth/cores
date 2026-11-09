# IOI causal controls and ACE overlap

| Model | Method | Rank | Keep AUC | Remove AUC | Flip AUC | Abs. cosine with ACE-1 |
|---|---|---:|---:|---:|---:|---:|
| GPT-2 Small | ACE selected | 7 | 1.000 | 0.951 | 0.012 | — |
| GPT-2 Small | ACE rank 1 | 1 | 0.997 | 0.936 | 0.360 | 1.000 |
| GPT-2 Small | PCA-1 | 1 | 0.876 | 1.000 | 1.000 | 0.170 |
| GPT-2 Small | Difference of means | 1 | 1.000 | 0.949 | 0.375 | 0.969 |
| GPT-2 Medium | ACE selected | 21 | 1.000 | 0.797 | 0.015 | — |
| GPT-2 Medium | ACE rank 1 | 1 | 0.814 | 0.999 | 0.963 | 1.000 |
| GPT-2 Medium | PCA-1 | 1 | 0.591 | 1.000 | 1.000 | 0.318 |
| GPT-2 Medium | Difference of means | 1 | 0.841 | 1.000 | 1.000 | 0.198 |
| GPT-2 Large | ACE selected | 17 | 1.000 | 0.985 | 0.015 | — |
| GPT-2 Large | ACE rank 1 | 1 | 1.000 | 0.999 | 0.967 | 1.000 |
| GPT-2 Large | PCA-1 | 1 | 0.959 | 1.000 | 1.000 | 0.073 |
| GPT-2 Large | Difference of means | 1 | 1.000 | 0.999 | 0.946 | 0.966 |
| LLaMA-3.1-8B | ACE selected | 15 | 1.000 | 0.998 | 0.002 | — |
| LLaMA-3.1-8B | ACE rank 1 | 1 | 0.803 | 1.000 | 1.000 | 1.000 |
| LLaMA-3.1-8B | PCA-1 | 1 | 0.954 | 1.000 | 1.000 | 0.336 |
| LLaMA-3.1-8B | Difference of means | 1 | 0.998 | 1.000 | 0.991 | 0.031 |
| Gemma-2-9B | ACE selected | 15 | 1.000 | 1.000 | 0.015 | — |
| Gemma-2-9B | ACE rank 1 | 1 | 1.000 | 1.000 | 0.997 | 1.000 |
| Gemma-2-9B | PCA-1 | 1 | 1.000 | 1.000 | 1.000 | 0.244 |
| Gemma-2-9B | Difference of means | 1 | 1.000 | 1.000 | 0.998 | 0.982 |
| Qwen-2.5-32B | ACE selected | 17 | 1.000 | 0.998 | 0.009 | — |
| Qwen-2.5-32B | ACE rank 1 | 1 | 0.902 | 1.000 | 1.000 | 1.000 |
| Qwen-2.5-32B | PCA-1 | 1 | 0.945 | 1.000 | 1.000 | 0.396 |
| Qwen-2.5-32B | Difference of means | 1 | 0.997 | 1.000 | 0.999 | 0.022 |
