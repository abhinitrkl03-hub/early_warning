# 📊 Financial Stress Early Warning System

An XGBoost-based machine learning system for predicting the probability of financial stress in companies using historical financial statement data and financial ratios.

The project combines financial data processing, feature engineering, temporal validation, machine learning, threshold optimization, and SHAP explainability into an interactive Streamlit application.

---

## 🚀 Project Overview

Financial distress can significantly affect companies, investors, lenders, and other stakeholders. The objective of this project is to develop an **Early Warning System (EWS)** capable of identifying companies that may experience financial stress in the following financial year.

The system uses historical company-level financial data to:

- Extract and clean financial information
- Engineer financial ratios and growth indicators
- Create a next-year financial stress target
- Compare multiple machine learning algorithms
- Perform temporal validation
- Tune the XGBoost model
- Optimize the classification threshold
- Explain model predictions using SHAP
- Provide predictions through a Streamlit web application

---

## 🎯 Objective

The primary objective is:

> **To develop an interpretable machine learning-based early warning system for identifying potential financial stress in companies.**

The model predicts:

```text
0 → Healthy
1 → Financially Stressed
