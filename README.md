# Data-Driven-Personalized-Educational-Content-Recommendation-System

view original **KT1 & Contents** [here](https://github.com/riiid/ednet)

```
project-root/
│
├── data/                # Processed datasets
│
├── src/                 # Python source files for models, utils, etc.
│   ├── recommender/     # TF-IDF, SVD, Hybrid models
│   ├── evaluation/      # Metrics and evaluation
│   └── utils/           # Preprocessing helpers
│
├── notebooks/           # Jupyter notebooks for EDA, prototyping
│   ├── 01_eda.ipynb
│   ├── 02_tf_idf.ipynb
│   └── 03_svd_hybrid.ipynb
│
├── api/                 # FastAPI backend
│   └── main.py
│
├── dashboard/           # Streamlit dashboard
│   └── enhanced_dashboard.py
│
├── tests/               # pytest unit tests
│
├── README.md
├── requirements.txt
└── .gitignore