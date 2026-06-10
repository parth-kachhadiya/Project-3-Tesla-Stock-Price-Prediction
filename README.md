# ⚡ TSLA Intelligence Dashboard

> A dark-themed, interactive Streamlit dashboard for Tesla stock analysis and multi-horizon LSTM price forecasting.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Dataset](#dataset)
- [Screenshots](#screenshots)
- [Disclaimer](#disclaimer)

---

## Overview

**TSLA Intelligence** is an end-to-end machine learning project that combines exploratory data analysis (EDA) with deep learning-based stock price forecasting. The dashboard is built with Streamlit and features a sleek dark UI styled around Tesla's brand identity.

Three separate LSTM models are trained for different prediction horizons — next day, up to 5 days, and up to 21 days — giving users a complete view from short-term signals to longer-term directional forecasts.

---

## Project Structure

```
├── data/
│   └── TSLA.csv                  # Historical Tesla stock data (2010–2020)
├── models/
│   ├── model_a.keras             # LSTM — 1-day forecast
│   ├── model_b.keras             # LSTM — up to 5-day forecast
│   ├── model_c.keras             # LSTM — up to 21-day forecast
│   ├── shared_scaler.pkl         # MinMaxScaler fitted on training data
│   └── metadata.json             # Model configs & performance metrics
├── notebooks/
│   ├── Tesla-Stock-Price-Prediction.ipynb   # Model training notebook
│   └── Univariate.ipynb                     # EDA & feature engineering
├── app.py                        # Streamlit dashboard
└── requirements.txt              # Python dependencies
```

---

## Features

### 📊 EDA — Exploratory Data Analysis

An interactive graph selector lets you pick any combination of 10 visualizations:

| # | Chart | Insight |
|---|-------|---------|
| 1 | **Adj Close Trend** | Full price history with annotated all-time high & low |
| 2 | **Yearly Boxplot** | Price distribution and outliers per year |
| 3 | **Volume vs Price Range** | Scatter with trend line & Pearson correlation |
| 4 | **Bullish vs Bearish** | Donut chart + yearly stacked bar sentiment breakdown |
| 5 | **Volume Spike vs Price** | Dual-axis chart with date-range filter; spikes highlighted in orange |
| 6 | **Monthly Heatmap** | Year × Month average price heatmap (RdYlGn scale) |
| 7 | **Correlation Heatmap** | Feature correlation matrix across OHLCV + engineered features |
| 8 | **Drawdown Analysis** | Rolling peak vs actual price + max drawdown annotation |
| 9 | **Rolling Volatility** | 30-day std dev with 75th percentile threshold overlay |
| 10 | **Quarterly Analysis** | 3-panel chart: avg price, avg volume, and QoQ return % |

A **KPI strip** at the top shows Total Days, Date Range, All-Time High, All-Time Low, and Max Drawdown at a glance.

---

### 🔮 Prediction — Multi-Horizon Forecasting

- Choose between **Model A**, **B**, or **C** from the sidebar
- Adjust forecast horizon with an interactive slider (up to 5 or 21 days)
- Three display styles: **Points**, **Line**, or **Smooth Line**
- **Animated forecast**: predictions appear day-by-day on the chart
- **Forecast results table**: shows price, daily change (↑/↓), and percentage move for each predicted day
- **Total change summary** at the bottom with color-coded direction

---

### 🧠 Model Details

- Architecture cards for each model (layer-by-layer breakdown)
- Performance metrics table with green highlights on best values
- R² comparison bar chart across all three models
- Training configuration summary (window size, epochs, batch size, validation split)

---

## Model Architecture

All three models share the same LSTM backbone but differ in their output layer size (direct multi-step forecasting strategy):

```
Input: (batch, 30, 1)   ← 30-day sliding window on Adj Close
  │
  ├── LSTM(64, return_sequences=True)
  ├── LSTM(32, return_sequences=False)
  ├── Dense(32, activation='relu')
  └── Dense(N)   ← N = 1 / 5 / 21 depending on model

Optimizer : Adam
Loss      : Mean Squared Error (MSE)
Epochs    : 100 (with early stopping)
Batch Size: 32
Val Split : 10%
Scaler    : MinMaxScaler (shared across all models)
```

| Model | Output Steps | Horizon |
|-------|-------------|---------|
| Model A | 1 | Next trading day |
| Model B | 5 | Up to 5 trading days |
| Model C | 21 | Up to 21 trading days |

---

## Model Performance

> Metrics evaluated on the held-out test set. Values are read from `models/metadata.json` and displayed live in the **Model Details** tab.

| Model | Horizon | R² | MAE | RMSE | MAPE% |
|-------|---------|-----|-----|------|-------|
| Model A | 1 Day | 0.97 |  7.70 | 13.34 | 2.44 |
| Model B | 5 Days | 0.89 | 15.03 | 20.92 | 4.86 |
| Model C | 21 Days | 0.84 | 15.65 | 19.65 | 4.40 |

> ℹ️ Fill in the actual values from your `metadata.json` after training. The dashboard's **Model Details** tab renders these live.

**Metric key:**
- **R²** — Coefficient of determination (higher is better; 1.0 = perfect)
- **MAE** — Mean Absolute Error in USD (lower is better)
- **RMSE** — Root Mean Squared Error in USD (lower is better)
- **MAPE%** — Mean Absolute Percentage Error (lower is better)

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Dashboard | `streamlit` |
| Deep Learning | `tensorflow` / `keras` |
| Data Processing | `pandas`, `numpy` |
| Visualization | `plotly`, `matplotlib`, `seaborn` |
| ML Utilities | `scikit-learn`, `joblib` |

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/tesla-stock-prediction.git
cd tesla-stock-prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the folder structure

Make sure these files exist before running the app:

```
data/TSLA.csv
models/model_a.keras
models/model_b.keras
models/model_c.keras
models/shared_scaler.pkl
models/metadata.json
```

> If you haven't trained the models yet, open and run `notebooks/Tesla-Stock-Price-Prediction.ipynb` first.

### 5. Launch the dashboard

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## Usage

| Action | How |
|--------|-----|
| Switch between EDA / Prediction / Model Details | Use the **NAVIGATE** radio in the sidebar |
| Choose which EDA charts to display | Use the **multiselect** dropdown in the EDA section |
| Filter volume spike chart by date range | Use the **Start / End date** pickers inside the chart card |
| Select a forecast model | Click the model radio button in the Prediction section |
| Set forecast horizon | Drag the **slider** (Model B: 1–5 days, Model C: 1–21 days) |
| Change forecast display style | Select Points / Line / Smooth Line above the chart |
| Use your own CSV | Upload via **"Upload TSLA CSV"** in the sidebar |

---

## Dataset

- **Source:** Yahoo Finance (TSLA historical daily OHLCV data)
- **Ticker:** `TSLA` (Tesla, Inc.)
- **Period:** 2010 – 2020
- **Frequency:** Daily (trading days only)
- **Columns:** `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`

**Engineered features computed at load time:**

| Feature | Description |
|---------|-------------|
| `Daily_Range` | `High − Low` per day |
| `Daily_Return` | Day-over-day % change in Adj Close |
| `Bullish` | `True` if `Close > Open` |
| `RollingStd30` | 30-day rolling standard deviation of Adj Close |
| `Drawdown` | % drawdown from rolling all-time high |
| `Vol_MA20` | 20-day rolling average volume |
| `Vol_Spike` | `True` if volume > 1.5× the 20-day average |

---

## Disclaimer

> **This project is built for educational and portfolio purposes only.**
> The LSTM models and forecasts produced by this dashboard are **not financial advice** and should not be used to make real investment decisions. Stock price prediction is inherently uncertain. Past performance does not guarantee future results.

---

<div align="center">
  Built with ⚡ for Tesla Stock Analysis &nbsp;|&nbsp; Powered by Streamlit & TensorFlow
</div>
