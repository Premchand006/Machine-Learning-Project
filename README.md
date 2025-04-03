# 🌞 Solar Irradiance Prediction using Machine Learning  

## 📌 Overview  
This project implements a **machine learning pipeline** to predict **solar irradiance** using regression models such as **Linear Regression** and **XGBoost**. The approach includes **data preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation** using various performance metrics.  

## 🚀 Features  
✅ **Data Preprocessing:** Handling missing values, scaling, and feature engineering  
✅ **Machine Learning Models:** Linear Regression, XGBoost  
✅ **Hyperparameter Optimization:** Using `RandomizedSearchCV`  
✅ **Model Evaluation:** RMSE, R² Score, MAE for performance assessment  
✅ **Visualization:** `matplotlib` & `seaborn` for graphical analysis  
✅ **Feature Importance Analysis:** Using **SHAP** for explainability  

---

## 📂 Project Structure  
```
📦 Solar_Irradiance_Prediction
│── 📜 final.ipynb              # Jupyter Notebook containing the full workflow
│── 📂 data/                    # Dataset directory (training & test data)
│── 📂 models/                  # Saved trained models (optional)
│── 📂 results/                 # Visualizations & analysis outputs
│── 📜 README.md                # Project documentation
```

---

## 🛠 Installation & Setup  
### 1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/your-repo/solar-irradiance-ml.git
cd solar-irradiance-ml
```

### 2️⃣ **Install Dependencies**  
Ensure you have **Python 3.8+** installed, then install the required packages:  
```bash
pip install -r requirements.txt
```
If `requirements.txt` is missing, install manually:  
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap jupyter
```

### 3️⃣ **Run the Jupyter Notebook**  
```bash
jupyter notebook final.ipynb
```

---

## 📊 Data Processing & Model Training  
1️⃣ **Load Dataset** – Read training and testing data from CSV  
2️⃣ **Preprocessing** – Handle missing values, standardize features  
3️⃣ **Feature Engineering** – Apply transformations and create new features  
4️⃣ **Model Selection** – Train **Linear Regression** and **XGBoost** models  
5️⃣ **Hyperparameter Tuning** – Use `RandomizedSearchCV` for optimization  
6️⃣ **Model Evaluation** – Compute RMSE, R², and visualize performance  
7️⃣ **Feature Importance** – Apply **SHAP** to interpret model decisions  

---

## 🎯 Model Evaluation Metrics  
The model is evaluated based on the following:  
✔ **Root Mean Squared Error (RMSE)** – Measures error magnitude  
✔ **R² Score (Coefficient of Determination)** – Indicates goodness of fit  
✔ **Mean Absolute Error (MAE)** – Measures absolute prediction errors  

Example Output (Replace with actual results):
| Model            | RMSE  | R² Score | MAE  |
|-----------------|------:|---------:|-----:|
| Linear Regression | 3.24  | 0.85     | 2.15 |
| XGBoost         | 2.58  | 0.92     | 1.75 |

---

## 📈 Visualizations  
The project includes **data insights and model evaluation graphs**, such as:  
📌 **Feature Correlations** – Understanding data relationships  
📌 **Actual vs. Predicted Scatter Plots** – Model accuracy representation  
📌 **SHAP Interpretability Plots** – Feature impact on predictions  

Example:  
```python
import shap
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
```
