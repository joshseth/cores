# SVA causal controls and ACE overlap

| Model | Method | Rank | Keep AUC | Remove AUC | Flip AUC | Abs. cosine with ACE-1 |
|---|---|---:|---:|---:|---:|---:|
| GPT-2 Small | ACE | 1 | 0.931 | 0.650 | 0.231 | 1.000 |
| GPT-2 Small | PCA-1 | 1 | 0.646 | 0.961 | 0.930 | 0.508 |
| GPT-2 Small | Difference of means | 1 | 0.975 | 0.646 | 0.094 | 0.955 |
| GPT-2 Small | Unembedding | 1 | 0.961 | 0.951 | 0.084 | 0.760 |
| GPT-2 Medium | ACE | 1 | 0.969 | 0.573 | 0.110 | 1.000 |
| GPT-2 Medium | PCA-1 | 1 | 0.665 | 0.972 | 0.969 | 0.271 |
| GPT-2 Medium | Difference of means | 1 | 0.989 | 0.560 | 0.044 | 0.969 |
| GPT-2 Medium | Unembedding | 1 | 0.978 | 0.910 | 0.035 | 0.807 |
| GPT-2 Large | ACE | 1 | 0.959 | 0.579 | 0.128 | 1.000 |
| GPT-2 Large | PCA-1 | 1 | 0.631 | 0.960 | 0.918 | 0.463 |
| GPT-2 Large | Difference of means | 1 | 0.978 | 0.559 | 0.042 | 0.971 |
| GPT-2 Large | Unembedding | 1 | 0.966 | 0.954 | 0.047 | 0.762 |
| LLaMA-3.1-8B | ACE | 1 | 0.778 | 0.683 | 0.395 | 1.000 |
| LLaMA-3.1-8B | PCA-1 | 1 | 0.588 | 0.943 | 0.934 | 0.502 |
| LLaMA-3.1-8B | Difference of means | 1 | 0.952 | 0.608 | 0.152 | 0.790 |
| LLaMA-3.1-8B | Unembedding | 1 | 0.947 | 0.886 | 0.079 | 0.328 |
| Gemma-2-9B | ACE | 1 | 0.879 | 0.573 | 0.241 | 1.000 |
| Gemma-2-9B | PCA-1 | 1 | 0.505 | 0.956 | 0.954 | 0.255 |
| Gemma-2-9B | Difference of means | 1 | 0.973 | 0.487 | 0.049 | 0.893 |
| Gemma-2-9B | Unembedding | 1 | 0.953 | 0.711 | 0.054 | 0.649 |
| Qwen-2.5-32B | ACE | 1 | 0.848 | 0.648 | 0.322 | 1.000 |
| Qwen-2.5-32B | PCA-1 | 1 | 0.518 | 0.954 | 0.951 | 0.285 |
| Qwen-2.5-32B | Difference of means | 1 | 0.964 | 0.603 | 0.123 | 0.890 |
| Qwen-2.5-32B | Unembedding | 1 | 0.954 | 0.912 | 0.062 | 0.456 |
