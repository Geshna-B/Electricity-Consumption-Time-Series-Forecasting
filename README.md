# Electricity Consumption Time Series Forecasting

Forecasting household electricity consumption using **Linear, Non-Linear, and Deep Learning models** including **ARIMA**, **GARCH**, **Custom LSTM**, and **Custom GRU** architectures on the UCI Household Power Consumption dataset.

---

## Project Overview

This project presents a comprehensive **time series forecasting framework** for household electricity consumption using classical statistical methods and advanced deep learning architectures.

The study benchmarks:

* **ARIMA(2,0,4)** for linear forecasting
* **GARCH(1,1)** for volatility modelling
* **Custom LSTM** with SE-Attention
* **Custom GRU** with dual-stream architecture

The models are trained and evaluated on nearly **2 million minute-level observations** collected over ~4 years from a French household.

---

## Team Members

* Geshna B
* Malavika S Prasad
* Vibhu Sanchana
* Asmi K

---

## Supervisor

Dr. Archuda A
Assistant Professor
Department of Artificial Intelligence
Amrita Vishwa Vidyapeetham, Coimbatore

---

## Dataset

This project uses the **Individual Household Electric Power Consumption** dataset from the UCI Machine Learning Repository.

### Dataset Information

* Source: UCI Machine Learning Repository
* Duration: December 2006 – November 2010
* Observations: ~2,075,259 minute-level records
* Features:

  * Global Active Power
  * Global Reactive Power
  * Voltage
  * Global Intensity
  * Sub-metering values

### Download Dataset

Download the dataset from:

[https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

### Dataset Placement

After downloading, place the dataset inside:

```bash
data/raw/
```

Expected file:

```bash
data/raw/household_power_consumption.txt
```

> Large raw datasets are not included in this repository due to GitHub file size limitations.

---

## Objectives

* Perform robust preprocessing and feature engineering on electricity consumption data
* Compare statistical and deep learning forecasting models
* Capture both linear and non-linear temporal dependencies
* Evaluate forecasting accuracy using multiple performance metrics
* Analyse residual behaviour and model limitations

---

## Models Used

### 1. ARIMA(2,0,4)

* Linear time series forecasting model
* Applied on stationary log-differenced daily series
* Selected using AIC/BIC grid search

### 2. GARCH(1,1)

* Models conditional variance and volatility clustering
* Applied on ARIMA residuals

### 3. Custom LSTM

* Dual-stream architecture
* SE-Attention mechanism
* Skip connections
* Dense prediction head

### 4. Custom GRU ★ Best Model

* Parameter-efficient architecture
* Superior generalisation performance
* 18% fewer parameters than LSTM

---

## Feature Engineering

A total of **78 engineered features** were created including:

* Temporal features
* Cyclical encodings
* Lag features
* Rolling statistics
* Exponential moving averages
* Interaction ratios
* Difference features
* Peak-load indicators

---

## Preprocessing Pipeline

* Missing value handling using forward/backward fill
* Minute → Hourly resampling
* Daily stationary transformation
* Feature scaling using:

  * StandardScaler
  * RobustScaler
* Sliding window sequence generation
* Blocked train/validation/test split

---

## Deep Learning Architecture

### Input Streams

#### Target Stream

* Sequence length: 168 hours
* LSTM/GRU layers
* SE-Attention block

#### Feature Stream

* 78 engineered features
* Parallel recurrent layers

#### Skip Connection

* Last known target value

### Dense Head

```text
256 → 128 → 64 → 1
```

---

## Results

| Model        | RMSE    | MAE     | R²     |
| ------------ | ------- | ------- | ------ |
| ARIMA(2,0,4) | 0.1288* | 0.0910* | -      |
| LSTM Custom  | 0.5376  | 0.3640  | 0.6482 |
| GRU Custom ★ | 0.5132  | 0.3419  | 0.6793 |

* Metrics on stationary log-differenced scale.

### Best Performing Model

✅ **Custom GRU**

* RMSE: 0.5132 kW
* MAE: 0.3419 kW
* R²: 0.6793
* 18% fewer parameters than LSTM

---

## Residual Analysis

Residual diagnostics showed:

* Slight positive bias
* Right-skewed residual distribution
* Persistent autocorrelation
* Non-normal residual behaviour

This indicates opportunities for future improvement using:

* Transformer architectures
* External covariates
* Advanced attention mechanisms

---

## Project Structure

```bash
Electricity-Consumption-Time-Series-Forecasting/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── output/
│   ├── plots/
│   └── models/
│
├── src/
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Visualisations

The project includes:

* Forecast evaluation plots
* ARIMA residual diagnostics
* GARCH volatility analysis
* Seasonality analysis
* Model comparison visualisations

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Statsmodels
* ARCH
* Matplotlib
* Seaborn

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Geshna-B/Electricity-Consumption-Time-Series-Forecasting.git
```

Navigate into the project directory:

```bash
cd Electricity-Consumption-Time-Series-Forecasting
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run preprocessing:

```bash
python src/preprocessing.py
```

Train models:

```bash
python src/train.py
```

Evaluate models:

```bash
python src/evaluate.py
```

---

## Future Work

* Transformer-based forecasting models
* Multi-step probabilistic forecasting
* Real-time smart meter deployment
* Weather and tariff integration
* Online learning for concept drift adaptation

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

* UCI Machine Learning Repository
* Amrita School of Artificial Intelligence
* TensorFlow & Statsmodels communities
