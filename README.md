# Predicting-Cancer-using-Logistic-Regression
# Breast Cancer Prediction using Machine Learning

A machine learning project for predicting breast cancer diagnosis using **Logistic Regression**.  
This project uses data preprocessing, feature scaling, train-test splitting, and classification evaluation to build a predictive model.

---
## 📌 Project Overview
The goal of this project is to classify breast cancer tumors as:
- **Malignant (M)** → Cancerous
- **Benign (B)** → Non-cancerous

The notebook performs:
- Data loading and preprocessing
- Handling missing values
- Feature scaling using StandardScalar
- Train-test splitting
- Logistic Regression model training
- Model evaluation using accuracy and classification report
---
## 📂 Dataset

The dataset contains medical diagnostic measurements for breast cancer prediction.

Features include:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry
- Fractal dimension
- and more...

Target variable:

- `M` → Malignant
- `B` → Benign

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Seaborn
- Scikit-learn
- Matplotlib

---

## 📦 Libraries Required

Install the required libraries using:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

---

## 🚀 Workflow

### 1. Import Libraries

```python
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
```

---

### 2. Load Dataset

```python
df = pd.read_csv("data.csv")
```

---

### 3. Data Preprocessing

- Removed unnecessary columns:
  - `Unnamed: 32`
  - `id`
- Converted diagnosis labels:
  - `M → 1`
  - `B → 0`

```python
df.diagnosis = [1 if value=="M" else 0 for value in df.diagnosis]
```

---

### 4. Feature Scaling

```python
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)
```

---

### 5. Train-Test Split

```python
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.3, random_state=42
)
```

---

### 6. Model Training

```python
lr = LogisticRegression()
lr.fit(x_train, y_train)
```

---

### 7. Prediction & Evaluation

```python
y_pred = lr.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
```

Evaluation metrics used:

- Accuracy Score
- Precision
- Recall
- F1-score
- Classification Report

---

## 📊 Model Performance

The Logistic Regression model achieved high classification accuracy on the test dataset.

Example evaluation:

```python
print(classification_report(y_test, y_pred))
```

---

## 📈 Future Improvements

Possible future enhancements:

- Try advanced models:
  - Random Forest
  - XGBoost
  - SVM
  - Neural Networks
- Hyperparameter tuning
- Feature selection
- Deployment using Flask/Streamlit
- Build a web application interface

---

## 📁 Project Structure

```bash
├── cancer_prediction.ipynb
├── Cancer diesease data.csv
└── README.md
```

---

## 🎯 Learning Outcomes

This project demonstrates:

- Data preprocessing
- Feature engineering basics
- Machine learning workflow
- Logistic Regression implementation
- Model evaluation techniques

---

## 👨‍💻 Author

**Aditya Kumar**  
Master's Student in Statistics  

Interested in:
- Machine Learning
- Data Science
- Statistical Modeling
- Research

---

## ⭐ If You Like This Project

Give this repository a ⭐ on GitHub and share your feedback!
