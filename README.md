# ❤️ Cardiovascular Disease Prediction

## 📌 Overview

This project aims to predict the presence of cardiovascular disease using machine learning techniques based on patient health data such as age, blood pressure, cholesterol levels, and lifestyle factors. Early prediction can help in timely diagnosis and prevention of severe heart conditions.

---

## 📊 Dataset

* Total Records: 70,000
* Features include:

  * Age
  * Gender
  * Height & Weight
  * Blood Pressure (Systolic & Diastolic)
  * Cholesterol Level
  * Glucose Level
  * Smoking & Alcohol Intake
  * Physical Activity

---

## ⚙️ Machine Learning Models

The following models were implemented and compared:

* Logistic Regression (LR)
* Random Forest (RF)
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Decision Tree (DT)

---

## 📈 Results

| Model                  | Accuracy |
| ---------------------- | -------- |
| Logistic Regression    | ~73–75%  |
| Random Forest          | ~74–77%  |
| Support Vector Machine | ~75–78%  |
| K-Nearest Neighbors    | ~70–74%  |
| Decision Tree          | ~68–72%  |

✅ **Best Performing Model:** Support Vector Machine / Random Forest

---

## 🔍 Key Insights

* 📊 Cardiovascular risk increases significantly with age
* 🧪 Higher cholesterol levels strongly correlate with disease
* 💓 High blood pressure is a critical risk factor
* ⚖️ BMI plays an important role in predicting heart disease
* 🚶 Physical activity reduces the likelihood of cardiovascular disease

---

## 📊 Data Analysis & Visualization

* Distribution plots for age, BMI, and target variable
* Count plots for cholesterol and lifestyle factors
* Correlation heatmap to identify relationships between features
* Feature importance analysis using Random Forest

---

## 🛠️ Tech Stack

* Python
* Pandas & NumPy
* Scikit-learn
* Matplotlib & Seaborn

---

## 🎯 Conclusion

Machine learning models can effectively predict cardiovascular disease with reasonable accuracy. Among all models, SVM and Random Forest performed the best. The study highlights the importance of age, blood pressure, cholesterol, and BMI in determining heart disease risk.

---

## 🚀 Future Improvements

* Hyperparameter tuning (GridSearchCV)
* Use of advanced models like XGBoost
* Deployment as a web application (Streamlit)
* Inclusion of more real-world clinical data

---
