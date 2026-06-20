# **Developing a Large Language Model for Bangla: Pretraining and Fine-Tuning for NLP Applications**

Tasmin Jahan	(0242310005101552)

### **Abstract**

Large Language Models for Bangla have mostly used encoder-only Transformer architectures. Standard attention scales quadratically with sequence length, producing memory explosions and degraded retrieval at long contexts. This paper introduces BanglaGSG, a foundational model pretrained on a Bangla-English-Banglish corpus that combines linear and attention mechanisms to address this bottleneck. The architecture interleaves Gated DeltaNet-2 (GDN), Sliding Window Attention (SWA), and Grouped-Query Attention (GQA) at a 1:1:1 ratio across 12 layers. This design inherits GDN's linear-time inference alongside the context retrieval capacity of SWA and GQA. Training runs on consumer hardware (12 GB VRAM) using mixed-precision BF16 and FlashAttention-2.

### **Introduction**

Attention mechanisms accelerated LLM development broadly, but Bangla has lagged due to limited data and compute. That compute shortage affects inference too: slower and more memory-hungry than necessary. Prior models, including BanglaGPT, BanglaLlama, and TigerLLM, adapted standard Transformer architectures to Bangla, but all remain constrained by the quadratic time and memory cost of attention.

BanglaGSG addresses this by combining GDN's linear-time inference with the retrieval strengths of attention. The architecture uses a strict 1:1:1 interleaved pattern of GDN, SWA, and GQA across 12 layers, along with SwiGLU activations, RMSNorm, and Rotary Position Embeddings (RoPE). Training from scratch on 12 GB VRAM required FlashAttention-2 and BF16 throughout.

### **Prior Work**

| Ref. | Authors (Year) / Source | Methodology | Key Findings | Limitations |
| :---- | :---- | :---- | :---- | :---- |
| \[1\] | Bhattacharjee et al. (2022) / NAACL 2022 | Pretrained BERT-based models (BanglaBERT/BanglishBERT) on 27.5 GB Bangla corpus using the ELECTRA objective. | Outperforms multilingual models (mBERT, XLM-R) on downstream NLU tasks while being highly sample and compute-efficient. | As an encoder-only architecture, it is primarily designed for NLU and lacks autoregressive generative capabilities. |
| \[2\] | Salim et al. (2023) / ICICT4SD | Trained a language-specific GPT-2 model (BanglaGPT) on a 26.24 GB corpus (BanglaCLM) with BPE tokenization. | Achieves a perplexity of 2.86, outperforming LSTM-based sequence-to-sequence models and mGPT for Bangla text generation. | Restricted to older GPT-2 architecture; explicitly limited to a fixed context size of only 128 tokens, truncating longer documents. |
| \[3\] | Zehady et al. (2024) / arXiv | Continually pretrained LLaMA variants on the CulturaX Bangla subset and instruction-tuned on translated Alpaca/Orca datasets. | Outperforms Meta-LLaMA baselines on reasoning, Open QA, and literature tasks due to Bangla-focused tuning. | Pretrained on a modest corpus size strictly due to "computational and resource constraints" associated with the LLaMA architecture. |
| \[4\] | Nahin et al. (2025) / arXiv | Adapted Llama-3.2 (1B, 3B) via continual pretraining (\~37B tokens) and developed 5 benchmarking datasets. | The extended tokenizer significantly improves reasoning and physical commonsense capabilities compared to base models. | Explicitly reports that "performance on long contexts remains suboptimal" and cites heavy computational constraints during training/inference. |
| \[5\] | Raihan & Zampieri (2025) / arXiv | Continually pretrained LLaMA/Gemma models on a 10M token Bangla-TextBook corpus and 100K distilled Bangla instructions. | Emphasizing data quality over quantity, TigerLLM outperforms existing open-source Bangla LLMs across six standard benchmarks. | Constrained by a small pretraining corpus; notes that scaling to larger architectures is severely limited by computational constraints. |

### 

### **Thematic Analysis**

**Low-Resource LLMs:** Prior Bangla NLP work has relied on standard Transformer architectures throughout. BanglaBERT \[1\] established the BLUB benchmark via an ELECTRA encoder but has no generative capability. BanglaGPT \[2\] introduced a small GPT-2 decoder capped at 128 tokens. BanglaLlama \[3\], TituLLMs \[4\], and TigerLLM \[5\] applied continual pretraining to LLaMA variants but ran into severe computational limits. Each model carries the quadratic attention cost of a standard Transformer. BanglaGSG differs from this line of work by training from scratch on a Bangla-code mixed corpus using a GDN-SWA-GQA hybrid, making it the first Bangla foundation model to use linear attention alongside optimized softmax attention.

**Optimized Attention and Linear Mechanisms:** Gated DeltaNet-2 (GDN) \[6\] achieves linear-time autoregressive inference by decoupling erase and write operations, maintaining high representation quality. Sliding Window Attention (SWA), used in Longformer \[7\] and Mistral 7B \[9\], limits each token's attention to a fixed local window, reducing both compute and KV cache size. Grouped-Query Attention (GQA) \[8\] shares key and value heads across multiple query heads, cutting memory usage and decoding latency.

**Hybrid Architectures:** Bae et al. \[10\] analyzed hybrid architectures and found that interleaving linear mechanisms with standard attention combines linear-decoding efficiency with in-context retrieval. A balanced interleave achieved the best perplexity-to-throughput tradeoff. BanglaGSG adopts this finding directly: a 1:1:1 interleave of GDN, SWA, and GQA across 12 layers.

**Architectural Optimizations:** SwiGLU \[11\] improves gradient flow and representation quality with minimal parameter cost. RMSNorm \[12\] removes the mean-centering step of layer normalization, reducing compute while keeping training stable. RoPE \[13\] encodes relative position in a way that generalizes better to longer sequences than sinusoidal embeddings.

**Training & Inference Optimization:** BF16 mixed-precision training \[14\] halves memory and roughly doubles throughput. FlashAttention-2 \[15\] tiles attention computation in SRAM, removing memory bandwidth as a bottleneck and making GQA practical at longer sequence lengths. TurboQuant \[16\] shows that compressing the KV cache to 3 bits cuts memory sixfold while increasing inference throughput up to eightfold. Hoffmann et al. \[17\] established compute-optimal scaling laws, suggesting a 1:20 parameter-to-training-token ratio (Chinchilla optimal). BanglaGSG targets a 1:40 ratio, training on roughly twice that token count for its parameter size.

### **Conclusion**

Bangla language models trained on standard Transformer architectures face quadratic memory and latency costs in long-context settings. BanglaGSG uses a 1:1:1 interleaved architecture of GDN, SWA, and GQA across 12 layers, pairing GDN's linear time complexity with the token retrieval capacity of SWA and GQA.

### **References**

\[1\] A. Bhattacharjee, T. Hasan, W. Ahmad, K. S. Mubasshir, M. S. Islam, A. Iqbal, M. S. Rahman, and R. Shahriyar, "BanglaBERT: Language model pretraining and benchmarks for low-resource language understanding evaluation in Bangla," in *Findings Assoc. Comput. Linguist.: NAACL 2022*, Seattle, WA, USA, Jul. 2022, pp. 1318–1327.

\[2\] M. S. Salim, H. Murad, D. Das, and F. Ahmed, "BanglaGPT: A generative pretrained transformer-based model for Bangla language," in *Proc. Int. Conf. Inf. Commun. Technol. Sustain. Dev. (ICICT4SD)*, Dhaka, Bangladesh, 2023\.

\[3\] A. K. Zehady, S. R. Dipta, and N. I. S. A. Mamun, "BanglaLlama: LLaMA for Bangla language," arXiv:2410.21200, Oct. 2024\.

\[4\] S. K. Nahin et al., "TituLLMs: A family of Bangla LLMs with comprehensive benchmarking," arXiv:2502.11187, Feb. 2025\.

\[5\] N. Raihan and M. Zampieri, "TigerLLM: A family of Bangla large language models," arXiv:2503.10995, Mar. 2025\.

\[6\] A. Hatamizadeh, Y. Choi, and J. Kautz, "Gated DeltaNet-2: Decoupling erase and write in linear attention," arXiv:2605.22791, May 2026\.

\[7\] I. Beltagy, M. E. Peters, and A. Cohan, "Longformer: The long-document transformer," arXiv:2004.05150, Apr. 2020\.

\[8\] J. Ainslie et al., "GQA: Training generalized multi-query transformer models from multi-head checkpoints," arXiv:2305.13245, May 2023\.

\[9\] A. Q. Jiang et al., "Mistral 7B," arXiv:2310.06825, Oct. 2023\.

\[10\] S. Bae et al., "Hybrid architectures for language models: Systematic analysis and design insights," arXiv:2510.04800, Oct. 2025\.

\[11\] N. Shazeer, "GLU variants improve transformer," arXiv:2002.05202, Feb. 2020\.

\[12\] B. Zhang and R. Sennrich, "Root mean square layer normalization," in *Adv. Neural Inf. Process. Syst.*, vol. 32, Vancouver, Canada, Dec. 2019, pp. 12360–12371.

\[13\] J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu, "RoFormer: Enhanced transformer with rotary position embedding," *Neurocomputing*, vol. 568, p. 127063, Feb. 2024\.

\[14\] P. Micikevicius et al., "Mixed precision training," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, Vancouver, Canada, 2018\.

\[15\] T. Dao, "FlashAttention-2: Faster attention with better parallelism and work partitioning," arXiv:2307.08691, Jul. 2023\.

\[16\] A. Zandieh, M. Daliri, M. Hadian, and V. Mirrokni, "TurboQuant: Online vector quantization with near-optimal distortion rate," in *Adv. Neural Inf. Process. Syst.*, vol. 37, Vancouver, Canada, Dec. 2024, pp. 140589–140631.

\[17\] J. Hoffmann et al., "Training compute-optimal large language models," arXiv:2203.15556, Mar. 2022\.

