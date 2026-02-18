# SeCon-RAG: A Two-Stage Semantic Filtering and Conflict-Free Framework for Trustworthy RAG

[[Paper]](https://arxiv.org/abs/2510.09710) [[NeurIPS 2025 Poster]](https://neurips.cc/virtual/2025/poster/115589)

[Xiaonan Si], [Meilin Zhu]()<sup>1</sup>, [Simeng Qin]()<sup>1</sup>, [Lijia Yu]()<sup>2</sup>, [Lijun Zhang]()<sup>1</sup>, [Shuaitong Liu]()<sup>1</sup>, [Xinfeng Li]()<sup>1</sup>, [Ranjie Duan]()<sup>1</sup>, [Yang Liu]()<sup>1</sup>, [Xiaojun Jia]()<sup>1</sup>

<sup>1</sup>Institute of Software, Chinese Academy of Sciences,


## 🔥 NEWS

- **2025.10**: 🎉 **SeCon-RAG** has been accepted by **NeurIPS 2025**!
- **2025.10**: The paper is available on [arXiv](https://arxiv.org/abs/2510.09710).
- **2025.12**: We have released the code for SeCon-RAG.

## 📖 Abstract

Retrieval-augmented generation (RAG) systems enhance large language models (LLMs) with external knowledge but are vulnerable to **corpus poisoning** and **contamination attacks**, which can compromise output integrity. Existing defenses often apply aggressive filtering, leading to unnecessary loss of valuable information and reduced reliability in generation. 

To address this problem, we propose **SeCon-RAG**, a two-stage semantic filtering and conflict-free framework for trustworthy RAG:
1.  **Stage 1: Semantic & Cluster-based Filtering.** Guided by an **Entity-Intent-Relation Extractor (EIRE)**, this stage extracts entities, latent objectives, and relations to selectively add valuable documents into a clean retrieval database.
2.  **Stage 2: Conflict-Aware Filtering.** An EIRE-guided module analyzes semantic consistency between the query, candidate answers, and retrieved knowledge to filter out internal and external contradictions.

Extensive experiments across various LLMs and datasets demonstrate that SeCon-RAG markedly outperforms state-of-the-art defense methods.

## 🛝 Try it out!

### 🛠️ Installation

```bash
git clone [https://github.com/lanmei666/Secon-Rag.git](https://github.com/lanmei666/Secon-Rag.git)
cd Secon-Rag

conda create -n seconrag python=3.10
conda activate seconrag



pip install lmdeploy

pip install beir

pip install nltk

pip install rouge_score

pip install timm==0.9.2

cd TrustRAG

python demo.py