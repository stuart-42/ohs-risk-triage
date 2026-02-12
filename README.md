# ohs-risk-triage
# 🦺 OHS Incident Severity Triage: NLP Decision Support Tool
**MSc Computer Science (AI) Dissertation Project | Northumbria University**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_URL_HERE)

## 📌 Project Overview
This prototype demonstrates the application of Natural Language Processing (NLP) to predict the **Incapacitation (Time Away from Work)** resulting from workplace incidents. 

The tool is designed for the **Health and Social Care sector** to assist safety professionals in the rapid triage of accident book entries, helping prioritize investigation resources based on predicted severity patterns.

## 🧠 Methodology & AI Model
* **Architecture:** Fine-tuned `DistilBERT` (Transformer-based) sequence classifier.
* **Dataset:** Trained on a curated corpus of Health & Safety incident narratives (sourced from US OSHA 2018-2023).
* **Task:** Multi-class classification mapping text narratives to five severity levels based on recovery time (None, Minor, Moderate, Severe, Major).
* **Explainability (XAI):** Utilizes **SHAP (SHapley Additive exPlanations)** with the Permutation algorithm to provide word-level transparency for every prediction.



## ⚖️ Safety & Legal Framework
This tool is built as an **Augmented Intelligence** artifact, meaning it supports human decision-making rather than replacing it.
* **RIDDOR Alignment:** The model’s predictions of "Time Away" are mapped to UK RIDDOR (Reporting of Injuries, Diseases and Dangerous Occurrences Regulations) thresholds.
* **Human-in-the-Loop:** High-severity flags are designed as investigative leads for a Competent Person, not as final regulatory determinations.

## 🚀 Deployment & Installation
The application is deployed via **Streamlit Cloud**. To run locally:

1. Clone the repo: `git clone https://github.com/stuart-42/YOUR_REPO.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Launch app: `streamlit run app.py`

## 📊 Ethical Considerations & Limitations
* **Training Bias:** The model is trained on US data; cultural or clinical differences in reporting may exist when applied to the UK context.
* **Data Privacy:** This prototype is intended for use with anonymized narratives. Users should never input Personally Identifiable Information (PII).
* **Scope:** Performance is optimized for the Care sector; results in Construction or Manufacturing may be less accurate.

## 📬 Contact & Feedback
* **Author:** Stuart Clark (MSc AI Candidate)
* **LinkedIn:** [Connect with me](https://www.linkedin.com/in/stuart-clark-161340164)
* **Support:** [Buy me a coffee](https://www.buymeacoffee.com/YOUR_USERNAME)
