# Student Performance Predictor and Classifier

## Overview
This project is an end-to-end machine learning pipeline implemented in a Jupyter Notebook (`Project_ML.ipynb`). It analyzes a student performance dataset containing 500 student records to predict continuous final scores (Regression) and binary pass/fail outcomes (Classification). The project explores data cleaning, feature scaling, hyperparameter tuning via K-Fold Cross-Validation, and rigorous model evaluation.

---

## Dataset Features
The dataset (`student_performance.csv`) includes the following attributes:
- **student_id:** Unique identifier for each student
- **study_hours_week:** Total hours spent studying per week
- **attendance_pct:** Attendance percentage in classes
- **prev_gpa:** Student's previous Grade Point Average
- **sleep_hours:** Average daily sleep hours
- **assignments_done_pct:** Percentage of completed assignments
- **part_time_job:** Binary indicator if the student holds a part-time job (0 or 1)
- **final_score:** Continuous final score achieved (Target for Regression)
- **passed:** Binary indicator of whether the student passed (Target for Classification)

---

## Models Implemented

### 1. Regression Models (Predicting Final Score)
- **Linear Regression:** Evaluated on cleaned features with $R^2$ and RMSE metrics.
- **Polynomial Regression:** Tested across degrees 1 to 5 using 5-Fold Cross-Validation to determine the optimal degree (Degree 2 chosen).

### 2. Classification Models (Predicting Pass/Fail Status)
- **Logistic Regression:** Scaled features trained for binary classification.
- **Decision Tree Classifier:** Optimized using K-Fold CV and `GridSearchCV` to find the best maximum depth (Depth = 5).
- **K-Nearest Neighbors (KNN):** Tuned using cross-validation and grid search to find the optimal number of neighbors ($K = 8$).

### 3. Model Comparison Matrix
All models (including thresholded regression models) were evaluated using Accuracy, Precision, Recall, and F1 Score to ensure a fair performance comparison.

---

## Tech Stack
- **Language:** Python
- **Libraries:** 
  - `numpy` & `pandas` (Data manipulation & cleaning)
  - `matplotlib` (Data visualization)
  - `scikit-learn` (Model training, preprocessing, cross-validation, and metrics)
- **Environment:** Google Colab / Jupyter Notebook

---

## Setup and Usage Instructions

### Prerequisites
Make sure you have Python 3.x installed along with the required libraries:
```bash
pip install numpy pandas matplotlib scikit-learn