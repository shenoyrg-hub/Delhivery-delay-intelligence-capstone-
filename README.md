# Delhivery Regional Operations Analytics

An end-to-end analytics and machine learning pipeline for Delhivery's regional operations. This repository answers three core operational problems:
1. **Delivery Hours Estimation:** Predicting exact delivery duration at dispatch time.
2. **SLA Delay Risk Classification:** Proactively identifying parcels at risk of missing promised delivery times (Express 36h, Standard 60h, Economy 96h) to prioritize operational interventions.
3. **Hub Behavioral Segmentation:** Re-evaluating historical hub performance using data-driven clustering instead of static Tier labels.

---

## 📁 Repository Structure

```text
delhivery-operations-analytics/
│
├── data/                    # Raw logs, engineered features, and pipeline artifacts
├── notebooks/               # Exploratory data analysis & model experiments
├── src/
│   ├── __init__.py
│   ├── generator.py         # Synthetic shipment data generator
│   ├── part_a_cleaning.py   # Cleaning & feature engineering script
│   ├── part_b_regression.py # Regression model for delivery duration estimation
│   ├── part_c_classifier.py # SLA delay risk classifier
│   ├── part_d_hubs.py       # K-Means hub behavioral clustering
│   └── final_scoring.py     # End-to-end batch scoring & operational action trigger generator
│
├── .gitignore               # Ignored data files & environments
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
