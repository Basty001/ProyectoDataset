# Car Price Prediction — Data Preparation Pipeline 🚗

## Overview
This repository contains a professional-grade Data Engineering and Data Science preprocessing pipeline. It is designed to clean, audit, and transform the **Car Features and MSRP Dataset** (predicting vehicle suggested retail price) for future Machine Learning modeling.

## Project Structure
```text
car_price_project/
├── data/
│   ├── raw/                  # Original, immutable datasets
│   └── processed/            # Cleaned data ready for ML
├── docs/                     # Technical reports and documentation
├── notebooks/                # Jupyter notebooks for EDA and experimentation
├── src/                      # Source code for custom transformers and auditing
│   ├── __init__.py
│   ├── audit.py              # Checksum and integrity validation
│   ├── optimization.py       # Memory optimization and chunk processing
│   ├── transformers.py       # Custom Scikit-Learn pipeline steps
│   └── pipeline.py           # Pipeline construction
├── main.py                   # Master orchestration script
├── requirements.txt          # Python dependencies
└── README.md                 # Project instructions
```

## Setup Instructions
To replicate this environment locally, follow these steps in your terminal:

### 1. Clone the repository:
```bash
git clone <url-del-repositorio>
cd car_price_project
```

### 2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Execution
To run the automated ETL and preprocessing pipeline:
```bash
python main.py
```

---


## 👤 Autor

**Bastian Soto**  
Estudiante de Ciencia de Datos — Duoc UC  
[LinkedIn](https://www.linkedin.com/in/bastian-isaias-soto-gonz%C3%A1lez-2395202a5/) · [GitHub](https://github.com/basty200) · [Kaggle](https://www.kaggle.com/bastianisaias)

---
