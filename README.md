# Predictive Paradox: Electricity Demand Forecasting

This repository contains the machine learning pipeline developed for the **IITG.ai Predictive Paradox Recruitment Task**. The core objective of this project is to build a robust, short-term forecasting model to predict the next hour's electricity demand (`demand_mw`) on the national grid, ensuring grid stability and efficient energy management.

## 🎯 Project Objective
To forecast `demand_mw` at time $t+1$ using historical consumption, environmental factors, and macroeconomic indicators up to time $t$. Per the task constraints, the pipeline strictly utilizes classical machine learning architectures without relying on deep learning (LSTMs) or autoregressive packages (ARIMA/Prophet).

---

## 📊 Dataset Overview
The pipeline integrates three distinct datasets:
1. **PGCB Demand Data (`PGCB_date_power_demand.csv`)**: Hourly demand and power generation data, acting as the primary target variable source.
2. **Weather Data (`weather_data.csv`)**: Hourly environmental metrics including temperature, humidity, and cloud cover.
3. **Macroeconomic Data (`economic_full_1.csv`)**: Annual World Bank indicators regarding GDP, industrial growth, and electricity access.

---

## 🛠️ Data Preparation & Structural Integrity

Given the raw nature of the data, extensive preprocessing was implemented to handle anomalies and ensure structural integrity:

### 1. PGCB Demand Data Cleaning
* **Dimensionality Reduction:** Removed columns with an excessive number of missing values (`remarks`, `nepal`, `india_adani`, `wind`).
* **Timestamp Standardization:** Addressed irregular recording frequencies (e.g., 18:30) by flooring timestamps to the nearest hour.
* **Duplicate Resolution:** Resolved duplicate hour entries by aggregating them using the mean of their numeric values.
* **Missing Hour Injection:** Created a complete, continuous hourly date range. Missing hours were injected with `NaN` values to maintain strict chronological consistency.

### 2. Missing Value Imputation
* **Interpolation:** Continuous generation and demand features were filled using time-based linear interpolation.
* **Constant Imputation:** Missing `solar` generation values were safely filled with 0s.

### 3. Outlier Detection & Clipping
* **Identification:** Applied the Z-score method (threshold > 3) to identify extreme, undocumented spikes in grid demand and generation.
* **Mitigation:** Identified outliers were clipped using the Interquartile Range (IQR) method. Values falling outside the Q1 - 1.5*IQR and Q3 + 1.5*IQR boundaries were fenced to prevent model distortion while preserving underlying trends.

### 4. Macroeconomic Data Integration
* Filtered the World Bank dataset to isolate relevant features containing keywords like *GDP*, *Industry*, and *Electricity*, later manually narrowed down to the most critical indicators.
* Reshaped the dataframe from wide to long format, transforming indicators into columns.
* Missing annual data points were handled via interpolation, and the final economic dataset was joined to the hourly series by Calendar Year.

---

## ⚙️ Feature Engineering

Because non-sequential classical ML algorithms treat observations independently, the concept of "time" and "memory" was artificially engineered into the tabular dataset:

* **Lag Features:** * `demand_prev_hour`: The electricity demand from the immediately preceding hour (t-1).
  * `demand_prev_day_same_hour`: The demand exactly 24 hours ago (t-24), capturing daily consumption seasonality.
* **Cyclical Time Encoding:** Captured the cyclical nature of the week by encoding the "day of the week" using sine and cosine transformations (`sinx`, `cosx`). This gives the model a mathematical sense of consecutive days and weekend/weekday behavioral shifts.

---

## 🧠 Modeling Strategy

### Algorithm Selection
The chosen algorithm is **XGBoost (Extreme Gradient Boosting)**, an advanced tree-based regressor well-suited for tabular data and complex non-linear relationships.

### Hyperparameters
* `n_estimators`: 500 (Number of boosting rounds)
* `learning_rate`: 0.05
* `max_depth`: 6 (Controls tree complexity and prevents overfitting)

### Validation Rigor
To ensure zero data leakage, strict chronological separation was enforced:
* **Training Set:** All chronological data up to and including **2023**.
* **Hold-Out Test Set:** All data for the year **2024**.

---

## 📈 Results & Evaluation

The model was evaluated on the unseen 2024 test dataset using **Mean Absolute Percentage Error (MAPE)**.

* **Final Test MAPE:** 2.62%

An error rate of 2.62% indicates a highly accurate and stable forecasting model. On average, the model's predictions deviate from the actual grid demand by less than 3%, minimizing the risk of both wasted generation (overestimation) and load shedding (underestimation).

---

## 🚀 How to Run

1. Clone this repository.
2. Ensure you have the required datasets placed in the root directory.
3. Install dependencies:
   pip install pandas numpy scipy xgboost scikit-learn matplotlib
