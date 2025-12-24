# 🏦 Bank Intent Detection Demo

This project is a **Streamlit-based NLP demo application** that detects user intent in banking-related text using:

- ✨ **Spell & grammar correction** (T5 transformer)
- 🧠 **Sentence embeddings** (Sentence Transformers)
- 📊 **KNN-based intent classification** (scikit-learn)

The application is designed as a lightweight prototype for **intent detection systems** commonly used in:
- Call centers
- Chatbots
- Virtual banking assistants

---

## 🚀 Features

- Grammar & spelling correction using **T5-large-spell**
- Semantic embedding with **MiniLM**
- Fast intent prediction using a **KNN classifier**
- Streamlit UI for easy testing
- Cached model loading for performance

---

## 🧩 Architecture
      User Text
          ↓
  Spell / Grammar Correction (T5)
          ↓
  Sentence Embedding (MiniLM)
          ↓
  KNN Intent Classifier
          ↓
  Predicted Banking Intent


---

## 📦 Models Used

### 1️⃣ Spell & Grammar Correction
- Model: `ai-forever/T5-large-spell`
- Framework: Hugging Face Transformers

### 2️⃣ Sentence Embeddings
- Model: `sentence-transformers/all-MiniLM-L6-v2`

### 3️⃣ Intent Classification
- Algorithm: **K-Nearest Neighbors**
- Saved as: `knn_intent_model.joblib`

---

## 🎯 Supported Intents

The model predicts one of the following intents:

- abroad
- address
- app_error
- atm_limit
- balance
- business_load
- card_issues
- card_deposit
- direct_debit
- freeze
- high_value_payment
- joint_account
- latest_transactions
- pay_bill
