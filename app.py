import streamlit as st
import pandas as pd
import joblib

# ==================================
# LOAD MODEL DAN SCALER
# ==================================

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ==================================
# JUDUL APLIKASI
# ==================================

st.title("🏦 Sistem Peringatan Dini Churn Pelanggan")

st.write(
    "Aplikasi ini digunakan untuk memprediksi kemungkinan pelanggan akan berhenti menggunakan layanan bank (churn)."
)

# ==================================
# INPUT DATA PELANGGAN
# ==================================

credit_score = st.number_input(
    "Credit Score (300–900)",
    min_value=300,
    max_value=900,
    value=650
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age (Usia Pelanggan)",
    min_value=18,
    max_value=100,
    value=35
)

tenure = st.number_input(
    "Tenure (Lama Menjadi Nasabah dalam Tahun)",
    min_value=0,
    max_value=10,
    value=5
)

balance = st.number_input(
    "Balance (Saldo Rekening)",
    min_value=0.0,
    value=50000.0
)

products_number = st.number_input(
    "Products Number (Jumlah Produk Bank yang Digunakan)",
    min_value=1,
    max_value=4,
    value=1
)

credit_card = st.selectbox(
    "Memiliki Kartu Kredit?",
    ["Ya", "Tidak"]
)

active_member = st.selectbox(
    "Status Keaktifan Nasabah",
    ["Aktif", "Tidak Aktif"]
)

estimated_salary = st.number_input(
    "Estimated Salary (Estimasi Pendapatan)",
    min_value=0.0,
    value=50000.0
)

country = st.selectbox(
    "Country (Negara Asal Pelanggan)",
    ["France", "Germany", "Spain"]
)

# ==================================
# TOMBOL PREDIKSI
# ==================================

if st.button("Predict"):

    # Encoding
    gender = 1 if gender == "Male" else 0
    credit_card = 1 if credit_card == "Ya" else 0
    active_member = 1 if active_member == "Aktif" else 0

    country_Germany = 1 if country == "Germany" else 0
    country_Spain = 1 if country == "Spain" else 0

    # Data Input
    data = pd.DataFrame([[
        credit_score,
        gender,
        age,
        tenure,
        balance,
        products_number,
        credit_card,
        active_member,
        estimated_salary,
        country_Germany,
        country_Spain
    ]], columns=[
        "credit_score",
        "gender",
        "age",
        "tenure",
        "balance",
        "products_number",
        "credit_card",
        "active_member",
        "estimated_salary",
        "country_Germany",
        "country_Spain"
    ])

    # Scaling
    data_scaled = scaler.transform(data)

    # Prediksi
    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]

    # ==================================
    # HASIL
    # ==================================

    st.subheader("Hasil Prediksi")

    st.metric(
        label="Probabilitas Churn",
        value=f"{probability:.2%}"
    )

    if probability >= 0.70:
        st.error("🔴 Risiko Churn Tinggi")

    elif probability >= 0.40:
        st.warning("🟠 Risiko Churn Sedang")

    else:
        st.success("🟢 Risiko Churn Rendah")

    # Informasi tambahan
    if prediction == 1:
        st.write(
            "Model memprediksi pelanggan memiliki kecenderungan untuk berhenti menggunakan layanan bank."
        )
    else:
        st.write(
            "Model memprediksi pelanggan akan tetap menggunakan layanan bank."
        )