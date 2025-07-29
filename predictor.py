import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("loan_risk_model.pkl")
scaler = joblib.load("scaler.pkl")

# List of features used
features = ['Age', 'Monthly_Income', 'Loan_Amount', 'Loan_Tenure', 'Interest_Rate',
            'Collateral_Value', 'Outstanding_Loan_Amount', 'Monthly_EMI', 'Num_Missed_Payments', 'Days_Past_Due']

# Recovery strategy logic
def assign_recovery_strategy(risk_score):
    if risk_score > 0.75:
        return "Immediate legal notices & aggressive recovery attempts"
    elif 0.50 <= risk_score <= 0.75:
        return "Settlement offers & repayment plans"
    else:
        return "Automated reminders & monitoring"

def predict_and_assign(data):
    data_scaled = scaler.transform(data[features])
    risk_scores = model.predict_proba(data_scaled)[:, 1]
    data['Risk_Score'] = risk_scores
    data['Predicted_High_Risk'] = (risk_scores > 0.5).astype(int)
    data['Recovery_Strategy'] = data['Risk_Score'].apply(assign_recovery_strategy)
    return data
