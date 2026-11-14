# 🛰️ SpaceGuard AI: Spacecraft Telemetry Anomaly Detection

An end-to-end unsupervised machine learning system designed to monitor spacecraft health and detect sensor anomalies (such as sudden spikes and flatline malfunctions) in real-time telemetry data using the **NASA Telemanom (Channel P-1)** dataset.

---

## 📌 Problem Statement
Spacecraft subsystems continuously transmit high-frequency telemetry signals during missions. Manual monitoring is impractical, and delayed fault detection can lead to critical mission failures. Because real-world anomalies are rare and diverse, **SpaceGuard AI** uses unsupervised anomaly detection to learn normal operational behavior from healthy telemetry data and automatically flag abnormal intervals in unseen mission streams.

---

## 📂 Dataset Overview
* **Source:** NASA Telemanom — **Channel P-1**
* **Training Set:** `2,872` timesteps representing normal spacecraft operation (`26` features: `timestep`, `value`, and binary command channels `cmd_0` to `cmd_23`).
* **Testing Set:** `8,505` timesteps containing normal operations interspersed with ground-truth anomaly sequences (`labeled_anomalies.csv`).

---

## ⚙️ Machine Learning Pipeline

### 1. Ground Truth Mapping
* Parsed official NASA `anomaly_sequences` intervals for Channel `P-1` and mapped them into a binary target vector (`0` = Normal, `1` = Anomaly) for objective model benchmarking.

### 2. Feature Selection
* Conducted Exploratory Data Analysis (EDA) and unique-value profiling (`nunique()`).
* Eliminated zero-variance/constant command channels (`nunique <= 1`) and non-predictive index columns (`timestep`) to reduce noise and dimensionality.

### 3. Temporal Feature Engineering
To enable point-based models to capture sequential anomalies like sensor **flatlines** (where readings freeze at a constant value), rolling window features were engineered:
* `rolling_mean_15`: 15-step moving average of telemetry values.
* `rolling_std_15`: 15-step moving standard deviation (drops to `0` during flatline faults).
* `value_diff`: Absolute first-order difference between consecutive readings.

### 4. Preprocessing & Model Benchmarking
Features were standardized using `StandardScaler` and used to train three unsupervised anomaly detection algorithms:
* **Isolation Forest**
* **Local Outlier Factor (LOF)**
* **One-Class SVM**

Models were evaluated against NASA's ground-truth labels using **Precision**, **Recall**, and **F1-Score**, with the top-performing model exported via `joblib` (`spaceguard_model.pkl`).

### 5. Interactive Diagnostic Web App
Deployed an interactive monitoring dashboard using **Gradio** that allows operators to inspect specific mission timestep windows, view real-time diagnostic reports (`NORMAL` vs. `CRITICAL`), and visualize detected anomalies overlaid on the telemetry signal.

---

## 🚀 How to Run

1. **Install dependencies:**
   ```bash
   pip install datasets pandas numpy matplotlib seaborn scikit-learn pyarrow gradio huggingface_hub joblib
