# SC_6-NLP-Review-Sentiment-Classification


🏆 **Top 1 Winner – Kaggle Neural Networks Competition**
Organized by the Faculty of Computer and Information Sciences, Ain Shams University

## Project Overview

Our project is an **NLP review sentiment classification system** that maps user reviews into **five ordered sentiment classes**:

**Excellent, Very Good, Good, Bad, Very Bad**

This system helps platforms extract **structured insights from customer feedback**, enabling better decision-making and sentiment analysis.

---

## Techniques Explored

### 1. Text Preprocessing

We built extensive preprocessing pipelines to clean and normalize textual data:

* Regex-based cleaning and contraction expansion
* Lowercasing and punctuation normalization
* Removal of URLs, HTML tags, emails, and phone numbers
* POS-aware lemmatization
* Length limiting
* Stopword frequency and distribution analysis across sentiment classes

### 2. Imbalance Handling

To address class imbalance, we experimented with:

* Class-weighted loss functions
* Focal Loss variants

### 3. Embeddings

We experimented with multiple embedding approaches:

* **GloVe (100d, trainable)**
* **FastText (subword-based, OOV-aware)**
* **Contextual embeddings from RoBERTa**, used as features with BiGRU + Self-Attention (embedding extraction without full fine-tuning)

### 4. Transformer & Deep Learning Models

We evaluated multiple architectures:

* DeBERTa-v3 Large (fully fine-tuned)
* RoBERTa-large
* BERT-base
* DistilBERT-base
* Stacked BiLSTM + Custom Attention
* Multi-kernel CNN + BiLSTM
* Transformer Encoder
* BiGRU + Attention

### 5. Intermediate Layer Feature Aggregation

Instead of using only the final transformer layer, we experimented with **aggregating hidden states from the last k transformer layers** for richer feature representation.

### 6. Ensemble Strategies

We built ensembles across RoBERTa-large, BERT-base, and DistilBERT-base:

* Weighted averaging of predictions
* Stacked ensemble using Logistic Regression as a meta-learner

---

## Technologies & Frameworks

* **Deep Learning Libraries:** PyTorch, TensorFlow
* **NLP & Transformers:** Hugging Face Transformers
* **Python Tools:** scikit-learn, pandas, NumPy

---

## Repository & Usage

You can check the project repo [here](https://github.com/Rasha-Abd-El-Khalik/Neural-Network-2026).

