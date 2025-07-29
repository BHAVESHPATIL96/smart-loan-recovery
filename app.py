import streamlit as st
import pandas as pd
from predictor import predict_and_assign

st.title("📊 Smart Loan Recovery System")
st.write("Upload borrower data to get risk scores and recovery strategies.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.subheader("📥 Uploaded Data Preview")
    st.dataframe(data.head())

    try:
        result = predict_and_assign(data)
        st.subheader("🔍 Recovery Suggestions")
        st.dataframe(result[['Risk_Score', 'Recovery_Strategy'] + [col for col in result.columns if col not in ['Risk_Score', 'Recovery_Strategy']]].head())

        # Download link
        csv = result.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name='loan_recovery_predictions.csv',
            mime='text/csv'
        )
    except Exception as e:
        st.error(f"Error: {e}")
