# 🛒 Data-Driven Sales Forecasting for Supermart using XGBoost & LSTM Model

This project aims to forecast Supermart's sales using a hybrid approach that combines the strengths of **XGBoost** (for feature-based modeling) and **LSTM** (for sequence learning). The predictive system is designed to help in inventory planning, marketing decisions, and demand forecasting.

---

## 📌 Project Overview

Sales forecasting is critical for supply chain efficiency and profitability. This project leverages both **traditional machine learning** and **deep learning** techniques:

- **XGBoost** is used for feature-based learning (e.g., holidays, store type, promotions).
- **LSTM (Long Short-Term Memory)** networks capture temporal dependencies and trends in sales data.
- Built using **Python** and developed in **PyCharm IDE**.

---

## 🧠 Model Architecture

### 1. XGBoost
- Gradient boosting algorithm to predict near-term sales using engineered features.
- Handles categorical and missing data well.
- Used for modeling tabular, non-sequential aspects of the dataset.

### 2. LSTM
- Recurrent neural network tailored for time series forecasting.
- Captures temporal patterns in the sales data.
- Used for learning long-term trends and seasonality.

### 3. Hybrid Ensemble (Optional)
- Combines both model predictions (e.g., averaging or weighted blending) to improve forecast accuracy.

---

## 📂 Project Structure

```
Data-Driven-Salesforecasting-for-Supermart-using-XGBoost-LSTM-Model/
│
├── data/                     # Raw and processed datasets
│   └── supermart_sales.csv
│
├── notebooks/                # Jupyter notebooks for exploration & modeling
│   └── eda.ipynb
│   └── xgboost_model.ipynb
│   └── lstm_model.ipynb
│
├── src/                      # Source code
│   ├── preprocessing.py
│   ├── train_xgboost.py
│   ├── train_lstm.py
│   ├── hybrid_forecasting.py
│
├── models/                   # Saved models (pickle/h5)
│
├── results/                  # Output results, plots, metrics
│
├── requirements.txt          # Python dependencies
│
└── README.md                 # Project documentation
```

---

## 📊 Features Used

- Date-based features (month, week, day, year)
- Store information (type, location)
- Promotion and discount flags
- Rolling and lag sales metrics
- External factors (e.g., holidays, economic indicators - if available)

---

## 🛠 Technologies Used

- Python 3.x
- Pandas, NumPy, Matplotlib, Seaborn
- XGBoost
- TensorFlow/Keras (for LSTM)
- Scikit-learn
- PyCharm (IDE)

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/CSanthoshIT/Data-Driven-Salesforecasting-for-Supermart-using-XGBoost-LSTM-Model.git
cd Data-Driven-Salesforecasting-for-Supermart-using-XGBoost-LSTM-Model
```

### 2. Set up the Environment

```bash
pip install -r requirements.txt
```

### 3. Run the Models

**XGBoost:**

```bash
python src/train_xgboost.py
```

**LSTM:**

```bash
python src/train_lstm.py
```

**Hybrid Forecast (if implemented):**

```bash
python src/hybrid_forecasting.py
```

---

## 📈 Evaluation Metrics

- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

**Visualizations include:**

- Actual vs Predicted sales
- Feature importance (XGBoost)
- Loss curves (LSTM training)

---

## 🔮 Future Work

- Hyperparameter optimization (e.g., Optuna or GridSearch)
- Deployment via Flask/Streamlit dashboard
- Adding external data sources (weather, macroeconomics)
- Incorporating attention mechanisms in LSTM

---

## 🙌 Acknowledgements

- **Datasets:** Kaggle, internal dataset
- **Libraries:** XGBoost, TensorFlow/Keras, Scikit-learn

---

## 📬 Contact

**Santhosh C**  
Email: crazyeditz2k@gmail.com  
GitHub: [@CSanthoshIT](https://github.com/CSanthoshIT)
