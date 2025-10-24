# simple_app.py
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.title("🎓 Simple Student Performance Predictor")

# Load dataset
df = pd.read_csv("cstperformance01.csv")
# Drop rows with missing values
df = df.dropna()

# Encode all object (text) columns
for c in df.select_dtypes(include='object').columns:
    df[c] = df[c].astype('category').cat.codes

# Make sure everything is numeric
X = df.drop(columns=['Total_Score'])
y = df['Total_Score']

model = LinearRegression().fit(X, y)


# Basic preprocessing
label_cols = ['Gender', 'Department', 'Extracurricular_Activities', 
              'Internet_Access_at_Home', 'Parent_Education_Level', 'Family_Income_Level']
encoders = {c: LabelEncoder().fit(df[c]) for c in label_cols}
for c in label_cols:
    df[c] = encoders[c].transform(df[c])

X = df.drop(columns=['Total_Score'])
y = df['Total_Score']

model = LinearRegression().fit(X, y)

# User inputs
st.subheader("Enter Student Details:")
gender = st.selectbox("Gender", encoders['Gender'].classes_)
age = st.number_input("Age", 10, 25, 17)
attendance = st.slider("Attendance (%)", 0, 100, 80)
midterm = st.number_input("Midterm Score", 0, 100, 60)
final = st.number_input("Final Score", 0, 100, 70)
hours = st.slider("Study Hours per Week", 0, 60, 15)
if st.button("Predict Score"):
    data = {
        'Gender': encoders['Gender'].transform([gender])[0],
        'Age': age,
        'Department': 0,
        'Attendance (%)': attendance,
        'Midterm_Score': midterm,
        'Final_Score': final,
        'Assignments_Avg': 70,
        'Projects_Score': 65,
        'Study_Hours_per_Week': hours,
        'Extracurricular_Activities': 0,
        'Quizzes_Avg': 60,
        'Internet_Access_at_Home': 1,
        'Parent_Education_Level': 2,
        'Family_Income_Level': 1,
        'Stress_Level (1-10)': 5,
        'Sleep_Hours_per_Night': 7
    }

    input_df = pd.DataFrame([data])
    input_df = input_df.reindex(columns=X.columns, fill_value=0)  # ✅ match training columns

    pred = model.predict(input_df)[0]
    st.success(f"Predicted Total Score: {pred:.2f}")

