"""Streamlit dashboard for Customer Churn Prediction."""
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from src.model_utils import DATA_PATH, load_dataset, get_or_train_model

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")
st.title("📊 Customer Churn Prediction")
st.caption("Machine Learning dashboard • Random Forest Classifier")
st.info("Educational portfolio project. The included dataset is synthetic demo data, created to demonstrate the workflow; it is not real customer information.")

@st.cache_resource
def load_model_and_data():
    model, metrics = get_or_train_model()
    df, X, y = load_dataset()
    return model, metrics, df, X

try:
    model, metrics, df, features = load_model_and_data()
except Exception as exc:
    st.error(f"Could not load the project dataset/model: {exc}")
    st.stop()

# Summary cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers in dataset", f"{metrics['rows']:,}")
col2.metric("Churn rate", f"{metrics['churn_rate_percent']:.1f}%")
col3.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
col4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
st.caption("Model performance is measured on a held-out test split; it is not a guarantee of real-world performance.")

st.subheader("Explore the dataset")
left, right = st.columns(2)
with left:
    st.markdown("**Customer churn distribution**")
    churn_counts = df["Churn"].value_counts().reindex(["No", "Yes"]).fillna(0)
    st.bar_chart(churn_counts.rename(index={"No": "Stayed", "Yes": "Churned"}))
with right:
    st.markdown("**Churn by contract type**")
    contract_table = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
    if "Yes" in contract_table.columns:
        st.bar_chart(contract_table["Yes"].rename("Churn rate (%)"))
    else:
        st.write("No churn data available.")

with st.expander("Preview dataset"):
    st.dataframe(df.head(15), use_container_width=True)
    st.write(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

st.divider()
st.subheader("Predict churn for a customer")
st.write("Choose the customer's details, then press **Predict churn risk**.")
with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_label = st.selectbox("Senior citizen?", ["No", "Yes"])
        partner = st.selectbox("Has a partner?", ["Yes", "No"])
        dependents = st.selectbox("Has dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
    with c2:
        internet_service = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
    with c3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.slider("Monthly charges ($)", min_value=18.0, max_value=120.0, value=70.0, step=0.5)
        st.caption("Total charges will be estimated as tenure × monthly charges for this demo.")
    submitted = st.form_submit_button("Predict churn risk", type="primary", use_container_width=True)

if submitted:
    # Enforce valid no-service categories based on selected services.
    if phone_service == "No":
        multiple_lines = "No phone service"
    elif multiple_lines == "No phone service":
        multiple_lines = "No"
    services = [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
    if internet_service == "No":
        services = ["No internet service"] * len(services)
    else:
        services = ["No" if x == "No internet service" else x for x in services]
    online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies = services
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior_label == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": round(tenure * monthly_charges, 2),
    }])
    try:
        risk = float(model.predict_proba(customer)[0][1])
        pred = int(model.predict(customer)[0])
        st.subheader("Prediction result")
        a, b = st.columns([1, 2])
        with a:
            st.metric("Estimated churn probability", f"{risk:.1%}")
            st.progress(min(max(risk, 0.0), 1.0))
        with b:
            if pred == 1:
                st.warning("The model flags this customer as higher churn risk. Consider reviewing service issues and retention options.")
            else:
                st.success("The model predicts this customer is more likely to stay.")
        st.caption("This is a demo prediction from synthetic training data, not a validated business decision.")
    except Exception as exc:
        st.error(f"Could not make a prediction: {exc}")

st.divider()
st.caption("Project workflow: data preparation → preprocessing → Random Forest training → evaluation → customer-level prediction.")
