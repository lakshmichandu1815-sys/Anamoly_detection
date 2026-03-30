import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Title
st.title("📄 Automated Invoice Anomaly Detection System")

# Upload file
uploaded_file = st.file_uploader("Upload Invoice CSV File", type=["csv"])

if uploaded_file is not None:
    # Read file
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Uploaded Data")
    st.write(df)

    # Check required columns
    if 'amount' in df.columns and 'tax' in df.columns:

        # Model
        model = IsolationForest(contamination=0.1)
        df['anomaly'] = model.fit_predict(df[['amount', 'tax']])

        # Convert values
        df['anomaly'] = df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

        st.subheader("🔍 Detection Results")
        st.write(df)

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", csv, "results.csv", "text/csv")

        # Graph
        st.subheader("📈 Visualization")
        fig, ax = plt.subplots()
        colors = df['anomaly'].apply(lambda x: 1 if x == 'Normal' else 0)
        ax.scatter(df['amount'], df['tax'], c=colors)
        ax.set_xlabel("Amount")
        ax.set_ylabel("Tax")
        ax.set_title("Anomaly Detection")
        st.pyplot(fig)

    else:
        st.error("❌ CSV must contain 'amount' and 'tax' columns")