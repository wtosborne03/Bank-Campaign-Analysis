import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# load model
booster = xgb.Booster()
booster.load_model('campaign.model')
title = st.title('Marketing Success Predictor')

with st.container(border=True):
    # client input form
    description = st.text("Client details:")
    age_select = st.number_input(
        "Age", help="-1 if unknown", step=1, min_value=-1, max_value=120, value=50)
    job_select = st.selectbox("Job", ("admin.", "unknown", "unemployed", "management", "housemaid",
                                      "entrepreneur", "student", "blue-collar", "self-employed", "retired", "technician", "services"))
    married_select = st.radio(
        "Married", ["married", "divorced", "single"],)
    education_select = st.radio(
        "Education Level", ["unknown", "primary", "secondary", "tertiary"],)
    default_select = st.checkbox("Does the client have a default? (Debt)")
    balance_select = st.number_input("Client's Annual Balance")
    housing_select = st.checkbox("Does Client have Mortgage Credit")
    loan_select = st.checkbox("Does Client have Consumer Loans")
    contact_select = st.radio(
        "Contact Method", ["unknown", "telephone", "cellular"],)
    date_select = st.date_input(
        "Last Day they were contacted (This Year)", format="MM/DD/YYYY")
    duration_select = st.number_input(
        "Approximately how many seconds the Contact Method lasted", step=1)
    campaign_select = st.number_input(
        "How many times they have been contacted", step=1)

    contacted = st.checkbox("Contacted before campaign?")
    if (contacted):
        pdays_select = st.number_input(
            "Number of days that they were last contacted before the campaign", step=1)
        previous_select = st.number_input(
            "Number of times that they were contacted before the campaign", step=1)

    poutcome_select = st.radio("Result of last marketing campaign", [
        "unknown", "other", "failure", "success"],)
    submitted = st.button('Submit')

# predict success
if submitted:
    # encode form data
    if (contacted == False):
        pdays_select = -1
        previous_select = 0
    data = [[age_select, job_select, married_select, education_select, default_select, balance_select, housing_select, loan_select,
            contact_select, date_select.day, date_select.month, duration_select, campaign_select, pdays_select, previous_select, poutcome_select]]
    bank_data = pd.DataFrame(data, columns=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
                                            'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
                                            'previous', 'poutcome'])
    le = LabelEncoder()
    label_columns = ['default', 'housing', 'month', 'loan']
    for col in label_columns:
        bank_data[col] = le.fit_transform(bank_data[col])
    categorical_columns = ['job', 'marital',
                           'education', 'contact', 'poutcome']
    bank_data = pd.get_dummies(bank_data, columns=categorical_columns)

    # predict
    st.write("Client Data:", bank_data.head())
    Dclient = xgb.DMatrix(bank_data.values)
    pred = booster.predict(Dclient)
    st.write(f"Campaign Success Probability: {pred[0]}")

    if (pred[0] > 0.5):
        st.write(
            "The marketing campaign will most likely work on this client! 😀")
    else:
        st.write(
            "The marketing campaign probabily won't work. Try finding a better lead.")
