
import joblib
import pandas as pd

# Load the saved objects
preprocessor = joblib.load("loan_preprocessor.pkl")
model = joblib.load("loan_model.pkl")

print("Model loaded successfully!")

# Example applicant
applicant = {
    "person_age": 30,
    "person_gender": "female",
    "person_education": "Bachelor",
    "person_income": 80000,
    "person_emp_exp": 6,
    "person_home_ownership": "MORTGAGE",
    "loan_amnt": 12000,
    "loan_intent": "EDUCATION",
    "loan_int_rate": 9.5,
    "loan_percent_income": 12000 / 80000,
    "cb_person_cred_hist_length": 8,
    "credit_score": 740,
    "previous_loan_defaults_on_file": "No"
}

# Convert input into a DataFrame
applicant_df = pd.DataFrame([applicant])

# Apply preprocessing and predict
processed = preprocessor.transform(applicant_df)
prediction = model.predict(processed)[0]
probability = model.predict_proba(processed)[0][1]

status = "Approved" if prediction == 1 else "Not Approved"

print("Predicted status:", status)
print("Estimated approval probability:", round(probability * 100, 2), "%")
