# AAPL Stock Price Forecasting

Forecasting Apple Inc. (AAPL) closing stock prices using **ARIMA**, **SARIMA**, and **LSTM** models.  
This project compares traditional statistical forecasting with deep learning to understand how each performs on real-world financial time series data.

---

## Project Summary

Stock forecasting is a key challenge in financial data science.  
Here, historical AAPL data (2018–2025) is used to predict future closing prices.  
The notebook walks through:
- Building **ARIMA/SARIMA** models for trend-based forecasting  
- Training an **LSTM** network for sequence learning  
- Comparing performance and forecast accuracy

---

## Repository Structure

AAPL_Stock_Price_Forecasting/  
│  
├── Data/  
│   └── AAPL_historical.csv  
│  
├── Notebooks/  
│   └── stock_forecasting_pipeline.ipynb  
│  
└── README.md  

---

## Approach

1. **Data Preparation**
   - Pulled daily AAPL data using `yfinance`
   - Cleaned missing values and ensured business-day frequency

2. **ARIMA & SARIMA**
   - Performed grid search for optimal `(p, d, q)` parameters  
   - Selected **ARIMA(2, 2, 3)** (lowest AIC = 6165.47)  
   - Tested **SARIMA** for yearly seasonality — smoother but similar accuracy

3. **LSTM**
   - Scaled data with `MinMaxScaler`  
   - Used 60-day lookback windows  
   - Built a multi-layer LSTM model with dropout and early stopping  
   - Produced sharper, short-term accurate forecasts

4. **Evaluation**
   | Model | MAE | RMSE | Observation |
   |--------|------|------|--------------|
   | ARIMA(2, 2, 3) | 14.6883 | 17.9199 | Stable but lags during sharp changes |
   | LSTM | 6.2073 | 7.7119 | Tracks volatility and adapts faster |

---

## Key Insights
- **LSTM** outperformed ARIMA/SARIMA with ~2.5× lower RMSE.  
- **ARIMA/SARIMA** captured long-term trends but missed short-term swings.  
- Both flatten when forecasting beyond 6 months — highlighting uncertainty.  
- A **Hybrid ARIMA–LSTM** could combine interpretability and adaptability.

---

## Tools & Libraries
Pandas • NumPy • Statsmodels • TensorFlow/Keras • Scikit-learn • Matplotlib

---

## Future Work
- Implement a **Hybrid ARIMA–LSTM** model  
- Add market indicators like trading volume or sentiment  
- Explore **Transformer-based** time series architectures  
- Build a **Streamlit dashboard** for visualization

---

## Author

**Laith Waqas Mohammed**  
_Data Science Student | Financial Forecasting Enthusiast_  
Dublin, Ireland  
[LinkedIn](https://www.linkedin.com/in/laithwm)
