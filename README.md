# SeCon-RAG: A Two-Stage Semantic Filtering and Conflict-Free Framework for Trustworthy RAG

[[Paper]](https://arxiv.org/abs/2510.09710) | [[NeurIPS 2025 Poster]](https://openreview.net/forum?id=tTwZhy8JqY)

Xiaonan Si<sup>1*</sup>, Meilin Zhu<sup>2,3*</sup>, Simeng Qin<sup>4†</sup>, Lijia Yu<sup>5†</sup>, Lijun Zhang<sup>1†</sup>, Shuaitong Liu<sup>6</sup>, Xinfeng Li<sup>7</sup>, Ranjie Duan<sup>8</sup>, Yang Liu<sup>7</sup>, Xiaojun Jia<sup>7†</sup>

<sup>1</sup> *Institute of Software, Chinese Academy of Sciences, Beijing, China* <br>
<sup>2</sup> *Key Laboratory of System Software (Chinese Academy of Sciences) and State Key Laboratory of Computer Science, Institute of Software, Chinese Academy of Sciences, Beijing, China* <br>
<sup>3</sup> *University of Chinese Academy of Sciences, Beijing, China* <br>
<sup>4</sup> *Northeast University, China* <br>
<sup>5</sup> *Institute of AI for Industries, Nanjing, China* <br>
<sup>6</sup> *Southwest University, China* <br>
<sup>7</sup> *Nanyang Technological University, Singapore* <br>
<sup>8</sup> *Alibaba, China* <br>
<small><sup>*</sup> Equal contribution. &nbsp; <sup>†</sup> Corresponding authors.</small>

## 🔥 NEWS

- **2025.10**: 🎉 **SeCon-RAG** has been accepted by **NeurIPS 2025**!
- **2025.10**: The paper is available on [arXiv](https://arxiv.org/abs/2510.09710).
- **2025.12**: We have released the code for SeCon-RAG.

## 📖 Abstract

Retrieval-augmented generation (RAG) systems enhance large language models (LLMs) with external knowledge but are vulnerable to **corpus poisoning** and **contamination attacks**, which can compromise output integrity. Existing defenses often apply aggressive filtering, leading to unnecessary loss of valuable information and reduced reliability in generation. 

To address this problem, we propose **SeCon-RAG**, a two-stage semantic filtering and conflict-free framework for trustworthy RAG:
1. **Stage 1: Semantic & Cluster-based Filtering.** Guided by an **Entity-Intent-Relation Extractor (EIRE)**, this stage extracts entities, latent objectives, and relations to selectively add valuable documents into a clean retrieval database.
2. **Stage 2: Conflict-Aware Filtering.** An EIRE-guided module analyzes semantic consistency between the query, candidate answers, and retrieved knowledge to filter out internal and external contradictions.

Extensive experiments across various LLMs and datasets demonstrate that SeCon-RAG markedly outperforms state-of-the-art defense methods.

## 🛝 Try it out!

### 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/lanmei666/Secon-Rag.git
cd Secon-Rag

# Create and activate conda environment
conda create -n seconrag python=3.10 -y
conda activate seconrag

# Install required packages
pip install lmdeploy beir nltk rouge_score timm==0.9.2

# Run the demo
cd TrustRAG
python demo.py
```


## 🙏 Acknowledgement
```
Our code used the implementation of [Trust-RAG](https://github.com/HuichiZhou/TrustRAG).
```

## 📝 Citation and Reference
```
If you find this paper useful, please consider staring 🌟 this repo and citing 📑 our paper:

@inproceedings{si2025seconrag,
  title={SeCon-RAG: A Two-Stage Semantic Filtering and Conflict-Free Framework for Trustworthy RAG},
  author={Si, Xiaonan and Zhu, Meilin and Qin, Simeng and Yu, Lijia and Zhang, Lijun and Liu, Shuaitong and Li, Xinfeng and Duan, Ranjie and Liu, Yang and Jia, Xiaojun},
  journal={arXiv preprint arXiv:2510.09710},
  year={2025}
}
@inproceedings{si2025seconrag,
  title={SeCon-{RAG}: A Two-Stage Semantic  Filtering and Conflict-Free Framework for Trustworthy {RAG}},
  author={Xiaonan si and Meilin Zhu and Simeng Qin and Lijia Yu and Lijun Zhang and Shuaitong Liu and Xinfeng Li and Ranjie Duan and Yang Liu and Xiaojun Jia},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=tTwZhy8JqY}
}

```
