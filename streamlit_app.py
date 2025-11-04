
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "student_data.csv"

st.title("🎓 Student Performance Predictor")
st.write("Predict a student's final score using a Linear Regression model.")

# Load data
data = pd.read_csv(DATA)
features = ["Hours_Studied", "Attendance", "Assignments_Submitted", "Previous_Score"]
target = "Final_Score"

X = data[features]
y = data[target]

# Simple train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline
preprocessor = ColumnTransformer([("num", StandardScaler(), features)])
model = Pipeline([("prep", preprocessor), ("reg", LinearRegression())])
model.fit(X_train, y_train)

# Show quick R2
r2 = r2_score(y_test, model.predict(X_test))
st.metric("R² on hold-out data", f"{r2:.3f}")

# Inputs
st.subheader("Enter student details")
h = st.slider("Hours studied per week", 0.0, 30.0, 12.0, 0.5)
a = st.slider("Attendance (%)", 40.0, 100.0, 90.0, 1.0)
asg = st.slider("Assignments submitted (0-10)", 0, 10, 8, 1)
prev = st.slider("Previous score", 0.0, 100.0, 72.0, 1.0)

# Predict
if st.button("Predict final score"):
    inp = pd.DataFrame({
        "Hours_Studied": [h],
        "Attendance": [a],
        "Assignments_Submitted": [asg],
        "Previous_Score": [prev]
    })
    pred = model.predict(inp)[0]
    st.success(f"Predicted Final Score: **{pred:.2f}** / 100")

# Show data preview
with st.expander("See training data preview"):
    st.dataframe(data.head(20))
