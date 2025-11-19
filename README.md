# Automatic Time Complexity Predictor using Fine-Tuned CodeT5 & LoRA

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red) ![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow) ![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange)

## 📌 Project Overview
This project implements a **data-driven Transformer-based framework** to automatically predict the **worst-case time complexity** of Python code snippets.

Addressing the limitations of manual analysis and the ambiguity of real-world code, this model treats complexity analysis as a **multi-class classification problem** across 9 distinct complexity classes ranging from $O(1)$ to $O(n!)$.

The solution leverages **Salesforce CodeT5**, a pre-trained programming language model, fine-tuned using **Low-Rank Adaptation (LoRA)** for memory-efficient training. It achieves **87.5% testing accuracy**, significantly outperforming baseline sequence-to-sequence approaches.

## 🚀 Key Features
* **9-Class Classification:** Predicts complexities including $O(1), O(n), O(n^2), O(\log n), O(n \log n), O(\sqrt{n}), O(n!), O(2^n),$ and $O(n^3)$.
* **Parameter Efficient Fine-Tuning (PEFT):** Utilizes **LoRA** to fine-tune only **~0.7%** of the total parameters (rank $r=16$), enabling training on consumer-grade GPUs (T4).
* **AST Feature Engineering:** Enhances the model's understanding of code structure by extracting features like loop depth, recursion count, and matrix patterns using Python's **Abstract Syntax Tree (AST)**.
* **Data Augmentation:** Implements semantic-preserving augmentation (variable renaming, function wrapping) to balance the dataset.

## 🛠️ Tech Stack
* **Language:** Python 3.11
* **Base Model:** `salesforce/codet5-base` (Encoder-Decoder Transformer)
* **Libraries:**
    * `transformers` (Hugging Face)
    * `peft` (LoRA implementation)
    * `datasets`, `scikit-learn`
    * `astor` (AST manipulation)
* **Hardware Used:** Google Colab (Tesla T4 GPU).

## 📊 Dataset & Preprocessing
* **Source:** 1008 manually curated Python snippets from GeeksForGeeks, LeetCode, and Rosetta Code.
* **Augmentation:** Expanded to **1548 samples** to address class imbalance (specifically for $O(n!)$ and $O(2^n)$).
* **Input Format:** Code snippets are cleaned, normalized, and prefixed with the prompt: `Analyze time complexity: {code}`.
* **Tokenizer:** Truncated to 320 tokens to ensure efficiency.

## 🧠 Methodology: LoRA Configuration
To prevent overfitting and reduce memory usage, the model was fine-tuned using LoRA with the following hyperparameters:
* **Rank (r):** 16
* **Alpha:** 32
* **Dropout:** 0.05
* **Target Modules:** Query (`q`), Value (`v`), Key (`k`), Output (`o`) projection layers.
* **Trainable Parameters:** ~892,000 (vs 125M total parameters).

## 📈 Performance Results
The model was trained for 15 epochs using the AdamW optimizer.

| Metric | Score |
| :--- | :--- |
| **Training Accuracy** | **91.7%** |
| **Testing Accuracy** | **87.5%** |
| **Inference Speed** | ~3.7 samples/sec |

**Class-wise Performance Highlights:**
* $O(1)$: **100% Accuracy**
* $O(n^3)$: **100% Accuracy**
* $O(\sqrt{n})$: **96.3% Accuracy**
* $O(2^n)$: **92.3% Accuracy**

## 💻 Installation & Usage

### 1. Install Dependencies
```bash
pip install transformers datasets peft accelerate bitsandbytes astor
