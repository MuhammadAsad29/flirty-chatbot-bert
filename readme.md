# 💬 Flirty Chatbot (BERT-Based Text Classifier)

A deep learning project that detects **flirty vs. non-flirty messages** using a fine-tuned **BERT model**. This project combines two labeled text datasets, preprocesses them, trains multiple baselines and transformer models, and provides both notebook-based and Streamlit-based inference interfaces.

---

## 📂 Project Structure

```
Flirty-chatbot/
├── flirty_chatbot.ipynb        # Full training + evaluation notebook (Colab-compatible)
├── flirty_chatbot.py           # Streamlit app for interactive predictions
├── requirements.txt            # All required libraries with pinned versions
├── flirty_model/               # Fine-tuned BERT model files (weights, config, tokenizer)
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── special_tokens_map.json
├── train_combined.csv          # Combined training dataset (balanced)
├── val_combined.csv            # Combined validation dataset
├── test_combined.csv           # Combined testing dataset
├── .gitignore                  # Ignored files/folders for version control
└── README.md                   # This documentation file
```

---

## 🧠 Project Overview

This project detects whether a chat message contains **flirtatious language** or not. It fine-tunes a pre-trained BERT model on combined labeled datasets collected from multiple sources. The system is capable of classifying short text messages into two classes:

- **1 → Flirty** 😉
- **0 → Not Flirty** 😊

The model can be tested interactively through a **Streamlit web interface** or programmatically via scripts.

---

## 🚀 Features

- Dataset merging and cleaning pipeline (CSV, Parquet, JSON supported)
- Text preprocessing (URL removal, mentions, HTML tags, normalization)
- Baseline model using TF-IDF + Logistic Regression
- Transformer model: **BERT fine-tuning** for sequence classification
- GPU-accelerated training on Google Colab
- Evaluation with precision, recall, F1-score, and accuracy
- Streamlit-based inference app for easy interaction
- Ready-to-deploy structure for local or cloud environments

---

## 📊 Dataset Preparation

Two datasets were provided in ZIP format (`dataset1.zip`, `dataset2.zip`).
Each contained `train`, `validation`, and `test` files in CSV and Parquet formats.

After extraction and inspection:

```
Dataset1 Shapes:
Train1: (1584, 3) Validation1: (212, 3) Test1: (318, 3)

Dataset2 Shapes:
Train2: (5550, 3) Validation2: (212, 3) Test2: (318, 3)
```

After combination and cleaning:

```
Train: (7134, 2)
Validation: (424, 2)
Test: (636, 2)
```

Label Distribution:

```
Train: 0 → 3955, 1 → 3179
Validation: 0 → 212, 1 → 212
Test: 0 → 318, 1 → 318
```

---

## ⚙️ Model Training Pipeline

### 🪜 Steps

1. **Data Cleaning:** URL, mentions, HTML tag removal using regex.
2. **Preprocessing:** Tokenization, lowercasing, trimming, whitespace normalization.
3. **Exploration:** Checked label balance and message length distribution.
4. **Baseline Model:** TF-IDF + Logistic Regression → ~73% accuracy.
5. **BERT Model:** Fine-tuned `bert-base-uncased` with Hugging Face Transformers.
6. **Evaluation:** Accuracy, precision, recall, and F1 across validation and test sets.
7. **Deployment:** Saved fine-tuned model to `flirty_model/` directory.

### 🧩 Model Configuration

```python
learning_rate = 5e-5
num_train_epochs = 4
batch_size = 32
max_length = 150
optimizer = AdamW
loss_fn = CrossEntropyLoss
```

### 📈 Example BERT Evaluation Results

```
Epoch | Train Loss | Val Loss | Accuracy | Precision | Recall | F1
---------------------------------------------------------------
1     | 0.1469     | 0.5713   | 0.8349   | 0.7840    | 0.9245 | 0.8484
2     | 0.0409     | 0.7719   | 0.8632   | 0.8468    | 0.8867 | 0.8663
3     | 0.0135     | 0.8763   | 0.8726   | 0.8623    | 0.8867 | 0.8744
4     | 0.0098     | 0.9192   | 0.8584   | 0.8333    | 0.8962 | 0.8636
```
📁 Model Files & Access

The flirty_model folder is not included in this repository due to file size limits.
If you wish to run inference using the pre-trained model, you can download it directly from Hugging Face Hub:

➡️ Pretrained Model: M-Asad29/flirty-model-bert

Use it in your code as:

from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("M-Asad29/flirty-model-bert")
tokenizer = BertTokenizer.from_pretrained("M-Asad29/flirty-model-bert")


If you prefer, you can also train your own model using the provided notebook (flirty_chatbot.ipynb) and your own dataset, then save it in a local directory named flirty_model/ for inference.
---

## 🧪 Streamlit Web App

### 🔹 File: `app.py`

A lightweight, interactive web interface built with **Streamlit** to test messages.

#### Example Code

```python
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch, os

os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"

@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("flirty_model")
    tokenizer = BertTokenizer.from_pretrained("flirty_model")
    return model, tokenizer

model, tokenizer = load_model()

def get_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=150)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return "😉 That sounds flirty!" if pred == 1 else "😊 Just a normal message."

st.set_page_config(page_title="Flirty Chatbot", page_icon="💬")
st.title("💬 Flirty Chatbot (BERT)")

msg = st.text_input("Type your message:")
if st.button("Check") and msg.strip():
    st.write("🤖 Bot:", get_response(msg))
```

Run locally with:

```bash
streamlit run app.py
```

---

## 📦 Requirements

Key dependencies:

```
torch>=2.5.0
transformers>=4.35.0
streamlit>=1.30.0
evaluate
scikit-learn
pandas
numpy
matplotlib
tqdm
```

All exact pinned versions are available in `requirements.txt`.

Install everything with:

```bash
pip install -r requirements.txt
```

---

## 🧰 Technologies Used

- **Language:** Python 3.10+
- **Frameworks:** PyTorch, Transformers (Hugging Face)
- **Web Interface:** Streamlit
- **Tools:** pandas, numpy, tqdm, scikit-learn
- **Training:** Google Colab GPU (Tesla T4)

---

## 📊 Evaluation Metrics Explained

- **Precision:** Fraction of predicted flirty messages that were correct.
- **Recall:** Fraction of actual flirty messages the model correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.
- **Accuracy:** Overall percentage of correctly predicted messages.

Higher scores across all metrics represent better model performance.

---

## 🧩 Example Predictions

| Input Message                      | Model Prediction |
| ---------------------------------- | ---------------- |
| "You look stunning tonight."       | 😉 Flirty        |
| "Let's finish the project report." | 😊 Not Flirty    |
| "Can’t stop thinking about you."  | 😉 Flirty        |
| "See you at the meeting."          | 😊 Not Flirty    |

---

## 📁 Model Files

The `flirty_model` directory includes all the files required for loading and inference:

```
config.json
pytorch_model.bin
tokenizer.json
tokenizer_config.json
vocab.txt
special_tokens_map.json
```

These are auto-loaded by Hugging Face's `from_pretrained()` API.

---

## 🧠 Future Enhancements

- Integrate emotional context detection.
- Fine-tune on multilingual datasets.
- Deploy on Hugging Face Spaces or Streamlit Cloud.
- Add response generation for conversational flow.

---

## 🧾 Author

**Muhammad Asad**
*AI & Software Engineering Enthusiast*
Project Owner and Model Developer

---

## 🪪 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.

---

## 🖤 Acknowledgments

- Hugging Face Transformers team
- Google Colab (GPU runtime)
- Streamlit for interactive web deployment

---

### 💡 Citation

If you use this repository, please cite:

```
@misc{asad2025flirtychatbot,
  title={Flirty Chatbot: BERT-Based Flirt Detection},
  author={Muhammad Asad},
  year={2025},
  url={https://github.com/M-Asad29/flirty-chatbot-bert}
}
```
