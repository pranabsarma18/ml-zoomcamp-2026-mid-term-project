import pickle
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


# Load data
df = pd.read_csv("dataset/student-depression-full-train.csv")

y_train = df["depression"].values

X_train = df.drop(columns=[
    "id",
    "depression",
    "suicidal_thoughts"
])

# Identify feature types
numerical = X_train.select_dtypes(include="number").columns.tolist()
categorical = X_train.select_dtypes(exclude="number").columns.tolist()

# Numerical preprocessing
numerical_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    MaxAbsScaler()
)

# Categorical preprocessing
categorical_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical),
    ("categorical", categorical_pipeline, categorical)
])

# Complete pipeline
pipeline = make_pipeline(
    preprocessor,
    LogisticRegression(
        solver="liblinear",
        C=1.0,  # Replace with selected C
        max_iter=1000
    )
)

# Train
pipeline.fit(X_train, y_train)

# Training score
y_pred = pipeline.predict_proba

# Save
with open("model.bin", "wb") as f_out:
    pickle.dump(pipeline, f_out)

print("Model saved successfully")