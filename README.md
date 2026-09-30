🛡️ URL Phishing Detection & Prevention System

 ML/DL-powered phishing and malicious URL detection platform with real-time browser protection, a React + Bootstrap frontend dashboard, and a Flask REST API backend.

## Project Motivation

Browser black‑lists are purely reactive and miss brand‑new (“zero‑day”) phishing pages.  
This project builds a **feature‑driven, model‑based** alternative that learns patterns of malicious behaviour and flags dangerous links on the fly.

## Repository Layout

```
url_classification_system/
├── Dataset/                    # Final balanced dataset (CSV) & feature matrices
├── src/
│   ├── Feature_engineering/    # URL, domain & content feature extraction + selection
│   ├── Models_and_evaluations/
│   │   ├── Machine_Learning/   # Tree, RF, SVM, XGBoost, LightGBM
│   │   ├── CNN/                # Character-level CNN
│   │   └── LSTM/                # Bidirectional LSTM (production) + trainers
│   └── User_interface/
│       ├── back_end/      # Flask REST API for real-time inference
│       └── front_end/     # Browser Extension + Dashboard using "React + Bootstrap"
└── requirements.txt             # Core Python deps
```
 Pipeline

| Stage | Source / Method | Key Points |
|-------|-----------------|------------|
| **Malicious URLs** | [URLhaus](https://urlhaus.abuse.ch/) daily feed | ~265 k samples |
| **Legitimate URLs** | [Common Crawl CDX](https://commoncrawl.org) snapshot | Randomly sampled to balance classes |
| **Feature Extraction** | 30 + handcrafted features – URL / lexical – Domain / DNS – HTML & JS hints | Implemented in `feature_extraction_segment.py` |
| **Feature Selection** | Low‑variance pruning, Pearson/Spearman correlation, RF importance | `feature_selection.py` |
| **Train/Test Split** | 80 % stratified train (internal), 20 % hold‑out + **PhiUSIIL** external test | Ensures generalisation |

The prepared dataset (`Dataset/final_dataset_with_selected_features.csv`) is shipped for convenience.  

### Preprocessing Steps

1. Remove duplicates and null entries
2. Normalize URL encoding
3. Label encoding for multi-class classification
4. Stratified train/validation/test split (70/15/15)
5. Feature extraction pipeline applied to all splits

## ⚙️ Feature Engineering

Features are extracted across three dimensions for each URL:

 1. Lexical Features (URL String-Based)
 2. Host-Based Features (DNS / WHOIS)
 3. Content-Based Features (HTML Page)

---

```bash
python src/Dataset_construction/generate_dataset.py           # raw → balanced CSV
python src/Feature_engineering/feature_extraction_segment.py  # feature matrix
python src/Feature_engineering/feature_selection.py           # reduced feature set
````

---

## 🛠️ Tech Stack

### Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.11+ | Core runtime |
| Flask | latest | REST API server |
| Gunicorn | ~20.1.0 | Production WSGI server |
| pandas | ~2.2.3 | Data processing |
| numpy | ~2.2.4 | Numerical operations |
| scikit-learn | ~1.6.1 | ML models (RF, SVM, DT, LR) |
| tldextract | ~5.1.3 | TLD / subdomain parsing |
| urllib3 | ~2.3.0 | URL utilities |
| requests | ~2.32.3 | HTTP client for crawling |
| beautifulsoup4 | ~4.13.3 | HTML content parsing |
| warcio | ~1.7.5 | CommonCrawl WARC file reading |
| dnspython | ~2.7.0 | DNS record lookups |

### Frontend

| Technology | Purpose |
|------------|---------|
| React | UI framework |
| Bootstrap 5 | Styling and responsive layout |
| Axios | API communication |
| React Router | Page navigation |

### Browser Extension

| Technology | Purpose |
|------------|---------|
| JavaScript (Manifest V3) | Extension logic |
| Chrome Extensions API | Chrome / Edge support |
| WebExtensions API | Firefox support |
| HTML / CSS | Popup UI |

### ML / DL Models

| Model | Type | Library |
|-------|------|---------|
| Random Forest | Ensemble ML | scikit-learn |
| XGBoost | Gradient Boosting | xgboost |
| LightGBM | Gradient Boosting | lightgbm |
| SVM | Classical ML | scikit-learn |
| Decision Tree | Classical ML | scikit-learn |
| CNN (1D) | Deep Learning | torch |
| LSTM | Deep Learning | torch |

pip install flask flask_cors

---

### Back-end

cd src/User_interface\ back-end code
python main.py

### Browser-Extension

open chrome, go to Extension, then load unpacked, choose your extension file(browser-extension)

###  Front‑end (dashboard)

```bash
cd src/User_interface\ front-end code\ phishing-frontend
npm start

---

## 📡 API Reference

Base URL: `http://localhost:5000`
```

## 8. Reproducibility Checklist ✅

* **Deterministic splits** – `random_state=42` everywhere
* **Environment lockfile** – see `requirements.txt`
* **Seed logging** – trainers emit seeds to `training.log`
* **Model checkpointing** – best‐by‑F1 per fold, plus final ensemble weight

---

## 9. Security & Ethical Considerations

* **Responsible use** – The detector is for *defensive* security research.
  Misuse to profile benign domains or censor content is disallowed by the license.
* **Dataset license** – Original feeds are public‑domain; redistributed samples are anonymised.
* **False positives** – Always show users a warning rather than hard‑blocking access.

---

*Happy phishing‑hunting! 🕵️‍♂️*
