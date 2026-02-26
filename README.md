# 📈 2026 Quantitative Finance Engine (Stacked LSTM)

A production-grade algorithmic trading engine that utilizes **Stacked LSTM (Long Short-Term Memory)** networks to forecast NSE equity prices (e.g., Reliance.NS).

## 🏛️ System Architecture
- **Data Persistence Layer:** Automated ingestion of NSE market data into a dedicated **SQLite** vault using `yfinance`.
- **Feature Engineering Pipeline:** Implementation of multi-variate analysis including **MA50 (50-day Moving Average)** and daily volatility tracking.
- **Deep Learning Engine:** 3-layer Stacked LSTM with **Dropout (0.2)** layers to ensure robust generalization and prevent overfitting.

## 📂 Project Structure
- `/data`: Persistence layer for SQL databases (`stock_vault.db`).
- `/models`: Serialized trained model weights (`reliance_model.keras`).
- `/src`: Modular source code for Ingestion, Training, and Prediction.

## 🚀 Key Metrics
- **Performance:** Designed to handle temporal dependencies in high-dimensional financial data.
- **Core CSE Fundamentals:** Demonstrates proficiency in modular software architecture and version-controlled environments.