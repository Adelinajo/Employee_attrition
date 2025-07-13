#predict.py

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load full dataset
df = pd.read_csv(r"C:\Users\sugan\Downloads\Employee Attrition\Emp_dataset.csv")  

# Select only 6 features
selected_cols = ['Age', 'Gender', 'MonthlyIncome', 'JobSatisfaction', 'OverTime', 'YearsAtCompany']
X = df[selected_cols]
y = df['Attrition']  # Target column (make sure it's binary: 0 or 1)

# Encode categorical if needed
X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})
X['OverTime'] = X['OverTime'].map({'Yes': 1, 'No': 0})

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_scaled, y_train)

# Save artifacts
joblib.dump(model, "lmodel_col.pkl")
joblib.dump(scaler, "scaler_col.pkl")
joblib.dump(selected_cols, "features_col.pkl")