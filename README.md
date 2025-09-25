
# Urdu → Roman Transliteration (BiLSTM Seq2Seq) — Overview

This project builds a neural transliteration system that converts **Urdu script** to **Roman Urdu** using a **sequence-to-sequence** model: a **character-level BiLSTM encoder** for Urdu, a **Luong-attention LSTM decoder** for Roman output, and a **state-depth bridge** to align encoder/decoder layers. Targets use **WordPiece** tokenization (we also evaluated char-level targets), with special tokens for PAD/BOS/EOS saved inside checkpoints for stable decoding. The dataset comprises aligned Urdu–Roman lines curated from **Rekhta** ghazal/poetry content; we normalized Urdu glyph variants and **removed diacritics** from the Roman side, then shuffled and split the corpus **50/25/25** (train/valid/test). Training uses PAD-masked Cross-Entropy (optionally **label smoothing ε=0.1**), **AdamW**, gradient clipping, and ReduceLROnPlateau; evaluation reports **per-token NLL/Perplexity**, **BLEU**, and **Character Error Rate (CER)**. Our best checkpoint reaches **test NLL ≈ 1.60 (PPL ≈ 4.9)** with **CER ≈ 27%** on the held-out set.

Key challenges and how we addressed them: 
(1) **Tokenization/formatting noise** produced spaced punctuation like `- e -` that penalized metrics; we added a **post-processor** (collapsing spaced hyphens/dots) and optionally normalized references similarly at eval time. 
(2) **Train–inference mismatch** (incorrect decoder inputs / early EOS) was fixed by enforcing **`dec_inp = [BOS] + tgt[:-1]`** and ensuring distinct PAD/BOS/EOS IDs persisted in the checkpoint. 
(3) A **train–valid gap** emerged as models deepened; **label smoothing** and moderating **weight decay (~1e-5)** reduced over-confidence and improved generalization. 
(4) **Depth vs. convergence:** transliteration favors **shallower decoders (2–3 layers)** with dropout 0.25–0.30; we observed steadier validation gains than with deeper stacks. 
(5) **Rare/archaic lexemes** in Rekhta prompted hallucinations; **beam search with length penalty** plus output cleanup curtailed long, off-target strings. Together, these choices yield a practical, robust transliterator for poetry-style Urdu sourced from **Rekhta**
[Online Hosted app link](https://bilstmnlp.streamlit.app/)
