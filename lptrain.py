import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
def load_data():
 df = pd.read_csv(r"C:\Users\sugan\Downloads\Employee Attrition\Emp_dataset.csv") 
 df = df[df["PerformanceRating"].isin([3, 4])]
 df["PerformanceRating"] = df["PerformanceRating"].map({3: 0, 4: 1})
 return df

df = load_data()

# Select features and target
features = ['Education', 'JobInvolvement', 'JobLevel', 'MonthlyIncome', 'YearsAtCompany', 'YearsInCurrentRole']
X = df[features]
y = df['PerformanceRating']  

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression(class_weight='balanced',max_iter=1000)
model.fit(X_train, y_train)

# Save artifacts
joblib.dump(model, "lmodel_col.pkl")
joblib.dump(scaler, "scaler_col.pkl")
joblib.dump(features, "features_col.pkl")

print(df["PerformanceRating"].value_counts())
