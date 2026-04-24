<h1 align="center">Predictive Maintenance of Industrial Equipment using Machine Learning</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-FA0F00?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/Dataset-NASA%20CMAPSS-lightgrey?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

---

## What Does This Project Do?

This project predicts whether an industrial turbofan engine is approaching failure using real NASA sensor data. Raw sensor readings from 21 channels are preprocessed, reduced, and fed into four machine learning models — each trained and owned by a different team member. The final output is a Streamlit dashboard where you can input live sensor values and get an instant health status (Normal / Warning / Critical) along with model explanations and a visual health gauge.

The project covers the full ML pipeline — from raw data to deployed interface — and maps directly to course concepts across parametric methods, nonparametric methods, kernel machines, dimensionality reduction, and clustering.

---

## Results

| Model | F1 | Precision | Recall | ROC-AUC |
|-------|----|-----------|--------|---------|
| Logistic Regression | 0.846 | 0.757 | 0.960 | 0.992 |
| Naive Bayes | 0.838 | 0.742 | 0.961 | 0.992 |
| Decision Tree (pruned) | ~0.87 | ~0.85 | ~0.89 | ~0.95 |
| **Random Forest (PCA)** | **~0.93** | **~0.91** | **~0.94** | **~0.98** |

> All models evaluated on NASA CMAPSS FD001 test set with binary label: failure within 30 cycles = 1.

---

## Tech Used

- **Language:** Python 3.10+
- **ML Library:** Scikit-Learn
- **Data:** NASA CMAPSS FD001 ([Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps))
- **Dashboard:** Streamlit
- **Compute:** Google Colab (T4 GPU for Random Forest grid search)
- **Visualization:** Matplotlib, Seaborn
- **Model Persistence:** Joblib

---

## Directory Structure

```
Home/
├── models/
│   ├── lr_model.pkl
│   ├── gnb_model.pkl
│   ├── dt_model.pkl
│   ├── dt_importances.pkl
│   ├── rf_model.pkl
│   ├── pca_transform.pkl
│   ├── pca_scaler.pkl
│   ├── global_scaler.pkl
│   └── feature_cols.pkl
├── outputs/
│   ├── model_comparison.csv
│   ├── train_clean.csv
│   ├── test_clean.csv
│   ├── decision_rules.txt
│   ├── confusion_matrices.png
│   ├── eda_distributions.png
│   ├── learning_curves.png
│   └── roc_curve.png
├── ML_jackfruit_Model_Develop.ipynb
└── app.py
```

---

## How to Install and Run

**1. Clone the repo**
```bash
git clone https://github.com/arjsh16/predictive-maintenance
cd predictive-maintenance
```

**2. Install dependencies**
```bash
pip install scikit-learn streamlit pandas numpy matplotlib seaborn joblib statsmodels
```

**3. Get the dataset**

Download from [Kaggle — NASA CMAPSS](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) and place these 3 files in the root directory:
```
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

**4. Run the notebook**

Open `ML_jackfruit_Model_Develop.ipynb` in Google Colab or Jupyter. Run all cells top to bottom. This generates all `.pkl` and output files.

**5. Launch the dashboard**
```bash
streamlit run app.py
```

---

## Authors

This project was built as part of a Machine Learning course. Work was divided equally across the pipeline.

| Author | Contribution | GitHub |
|--------|-------------|--------|
| **Archi** | Data pipeline, EDA, preprocessing, RUL labeling, Logistic Regression, Naive Bayes, Streamlit input form | [@archiarchi11](https://github.com/archiarchi11) |
| **Arjun** | Decision Tree with pruning, IF-THEN rule extraction, feature importance, outlier detection (Isolation Forest + LOF), Streamlit explanation tab | [@arjsh16](https://github.com/arjsh16) |
| **Ansh** | PCA dimensionality reduction, Random Forest, health gauge visualization, final model comparison, Streamlit deployment | [@ansh-deshwal](https://github.com/ansh-deshwal) |

---

## Scope of Improvements

- **SVM with RBF kernel** was scoped out due to time — adding it would complete the Unit 3 kernel machines comparison
- **LSTM / GRU** on raw time-series data instead of snapshots would likely push F1 above 0.95
- **Real-time sensor streaming** via MQTT or Kafka to make the dashboard production-ready
- **Remaining Useful Life regression** instead of binary classification for more actionable predictions
- **Explainability** — integrating SHAP values across all models, not just Decision Tree
- **Multi-dataset support** — currently only FD001; extending to FD002/FD003/FD004 would test generalization

---

## License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute this code with attribution. See the [LICENSE](./LICENSE) file for details.

---

## Feel Free to Contact Us

Have questions, suggestions, or want to collaborate?

- Reach out via GitHub Issues on this repo
- [Archi](https://github.com/archiarchi11) · [Arjun](https://github.com/arjsh16) · [Ansh](https://github.com/ansh-deshwal)

> If you found this project useful, consider leaving a ⭐ on the repo!