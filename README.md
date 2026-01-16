# Bilingual Neural Translation (English ↔ Hindi)

This repository contains two end-to-end implementations of a **Neural Machine Translation (NMT)** system that translates sentences between **English and Hindi** using Transformer models.

The project begins with text cleaning, tokenization, train/val splitting, model building, and finally evaluation + prediction.

---

## Project Contents

| File | Description |
|------|-------------|
| `Transformers only.py` | Basic Transformer-based translation model (encoder + decoder, no self-attention extension). |
| `Transformers + Attention.py` | Expanded implementation with contextual **Multi-Head Attention** and improved decoding output. |

---

## Features

✔ Cleans and normalizes parallel English–Hindi text  
✔ Removes noise, symbols, code-like text, duplicates  
✔ Supports Unicode Devanagari (Hindi)  
✔ Builds vocabulary tokenizers  
✔ Trains a Transformer-based NMT model  
✔ Evaluates translation performance  
✔ Predicts translations using trained model  
✔ Demonstrates how attention improves context learning

---

## Model Architecture

### 1 **Transformers Only**
- Encoder–decoder Transformer
- Positional encodings
- Token embeddings
- Multi-head attention inside layers
- Trained using teacher forcing

### 2 **Transformers + Attention**
- All features from baseline model, plus:
- Explicit **attention heads**
- Better handling of long-range dependencies
- More accurate context mapping
- Clear improvements in prediction quality

---

## Technologies Used

- Python
- NumPy  
- Pandas  
- Matplotlib  
- TensorFlow / Keras  
- Scikit-Learn  
- Regular Expressions (text cleaning)

---

## Dataset

Parallel bilingual text, containing paired: English sentence → हिंदी अनुवाद

Dataset is preprocessed to remove:
- Symbols + special characters
- Technical words / code fragments
- Empty and duplicated lines

---

## ▶️ How to Run


pip install -r requirements.txt
python "Transformers only.py"
python "Transformers + Attention.py" 

Make sure your dataset paths are correct inside the scripts.



## 📝 Sample Output

After training, you can pass any English sentence to the model and receive a translated output.  
For example:
Input : "How are you?"
Output : "तुम कैसे हो?"


## Project Workflow

1 **Load parallel dataset (English ↔ Hindi)**  
2 **Clean & normalize text**  
   - remove special characters  
   - remove mixed-language code noise  
   - lowercasing + strip spaces  
3 **Split into train & validation sets**  
4**Tokenize English & Hindi vocabulary**  
5 **Build Transformer model**
   - Encoder (multi-head self-attention + feedforward)
   - Decoder (masked attention + cross-attention)
6 **Train using teacher forcing**
7 **Evaluate accuracy / validation loss**
8 **Predict and translate unseen sentences**

---

## Key Differences Between the Two Scripts

| Feature | Transformers Only | Transformers + Attention |
|--------|------------------|--------------------------|
| Basic encoder-decoder | ✔️ | ✔️ |
| Positional encoding | ✔️ | ✔️ |
| Token embeddings | ✔️ | ✔️ |
| Explicit multi-head attention interpretation | ❌ | ✔️ |
| Better long-context understanding | ❌ | ✔️ |
| Cleaner final predictions | ❌ | ✔️ |

---

## 📦 Requirements

To install dependencies:
pip install -r requirements.txt

Typical packages used include:
- tensorflow / keras
- numpy
- pandas
- sklearn
- matplotlib
- re (regex)

---

## Future Improvements

- Use subword tokenizers (SentencePiece / Byte Pair Encoding)
- Train with a larger corpus (OpenSubtitles, IIT Bombay dataset, Tatoeba)
- Deploy model as a REST API using FastAPI / Flask
- Convert to ONNX for lighter inference
- Build a small Streamlit demo app

## Learning Outcomes

- Understand Transformer internals (encoder/decoder blocks)
- Learn how attention captures word relationships
- Experience cleaning raw multilingual data
-Get hands-on practice building your own translation model

## Author
Rakesh Pathlavath







