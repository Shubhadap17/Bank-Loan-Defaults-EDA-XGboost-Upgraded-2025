# Bank Loan Default Prediction using EDA and XGBoost

This project analyses 50,000 bank loan applications to identify patterns associated with default risk, then builds a recall-focused XGBoost model to flag high-risk applicants. It follows a structured data science workflow using Python, pandas, seaborn and XGBoost.

---

## 🔍 Project Objective

To explore loan applicant data using EDA, handle data quality issues, and build a machine learning model that predicts whether an applicant is likely to default, so that risky applicants are flagged without rejecting capable ones.

---

## 📦 Dataset

- **Source file:** `Q_application_data.csv` (50,000 loan applications, 122 columns)
- **After cleaning:** 45,191 records, split 80/20 into 36,152 training and 9,039 test records
- **Class imbalance:** about 11 repaid loans for every default (8% defaulters)

---

## ✅ Workflow Summary

- **Steps 1–7:** Data loading, missing value treatment (columns with over 40% missing dropped; median and mode imputation for the rest) and IQR outlier removal (2,110 rows removed)
- **Steps 8–13:** Class imbalance analysis, univariate, segmented and bivariate analysis, and correlation insights
- **Steps 14–15:** Baseline XGBoost model training
- **Step 17:** Class imbalance handled using manual undersampling
- **Steps 18–19:** Retrained model on balanced data and evaluated on the original test set
- **Steps 20–21:** Hyperparameter tuning using RandomizedSearchCV

---

## 📊 Model Performance

| Model | Accuracy | Recall (Defaulters) | AUC | Defaulters Caught |
|-------|----------|---------------------|-----|-------------------|
| Baseline XGBoost | 92% | 3% | 0.70 | 22 of 710 |
| Balanced (undersampled) | 66% | 66% | 0.70 | 468 of 710 |
| **Final tuned model** | **69%** | **68%** | **0.74** | **482 of 710** |

The baseline model looked accurate but missed 97% of defaulters because of the class imbalance. The final model trades 23 points of accuracy for a 65-point gain in defaulter recall, which suits credit risk, where missing a defaulter costs a lender far more than reviewing an extra application.

---

## 🧠 Key Insights

- The features most correlated with default were `EXT_SOURCE_3`, `EXT_SOURCE_2` and `DAYS_BIRTH` (applicant age)
- Applicants with lower external credit scores were more likely to default
- Manual undersampling was used for simplicity and explainability
- Hyperparameter tuning raised AUC from 0.70 to 0.74 while keeping recall high

---

## 🛠 Tech Stack

- Python (pandas, NumPy)
- seaborn, matplotlib
- scikit-learn
- XGBoost
- Google Colab

---

## 📁 Files Included

- `loan_defaults_eda_XGBoost.ipynb`: Full Colab notebook with code and markdown
- `Q_application_data.csv`: Loan application dataset
- `Bank_Loan_Defaults_Report_Shubhada.pdf`: Project report
- `README.md`

---

## 🙋 Author

- **Shubhada Patil**
- [GitHub Profile](https://github.com/Shubhadap17)
- [LinkedIn Profile](https://linkedin.com/in/shubhada-s-patil)

---

## 📎 License

This project is intended for educational and demonstration purposes. Contributions and forks are welcome.
