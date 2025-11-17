import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("Heart Regression App - Streamlit")

# Upload dataset
st.write("Upload dataset kamu (CSV):")
uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data:")
    st.dataframe(df.head())

    # Pilih target (Y)
    st.write("### Pilih Kolom Target (HARUS numerik):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    target_col = st.selectbox("Pilih target:", numeric_cols)

    # Fitur (X)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ==========================================
    # FIX: BERSIHKAN SEMUA KATEGORIKAL
    # ==========================================
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = (
                X[col]
                .astype(str)                                # Paksa string
                .str.replace(r"[^A-Za-z0-9 ]", "", regex=True)  # Bersihkan karakter aneh
                .str.strip()
                .str.upper()
            )

    # ==========================================
    # One-hot encoding
    # ==========================================
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Cek apakah masih ada non-numeric
    not_numeric = X_encoded.select_dtypes(exclude=[np.number]).columns
    if len(not_numeric) > 0:
        st.error("Masih ada fitur non-numerik setelah encoding.")
        st.write("Kolom bermasalah:", list(not_numeric))
        st.stop()

    # ==========================================
    # Train-Test Split
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write("### Hasil Evaluasi Model")
    st.write(f"**MSE:** {mse:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")

    # Visualisasi Prediksi
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Nilai Aktual")
    ax.set_ylabel("Prediksi")
    ax.set_title("Prediksi vs Aktual")
    st.pyplot(fig)

    # ================================
    #        PREDIKSI MANUAL
    # ================================
    st.write("### Coba Prediksi Manual")

    user_input = {}

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            user_input[col] = st.number_input(
                f"Masukkan nilai untuk {col}",
                float(df[col].min()),
                float(df[col].max())
            )
        else:
            user_input[col] = st.text_input(f"Masukkan nilai untuk {col} (kategori)").upper().strip()

    if st.button("Prediksi"):
        # Convert ke dataframe
        input_df = pd.DataFrame([user_input])

        # Cleaning sama seperti X
        for col in input_df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                input_df[col] = (
                    input_df[col]
                    .astype(str)
                    .str.replace(r"[^A-Za-z0-9 ]", "", regex=True)
                    .str.strip()
                    .str.upper()
                )

        # One-hot encoding input
        input_encoded = pd.get_dummies(input_df)

        # Samakan kolom seperti training
        input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

        # Prediksi
        prediction = model.predict(input_encoded)
        st.success(f"Prediksi nilai: {prediction[0]:,.2f}")

else:
    st.info("Silakan upload file CSV terlebih dahulu.")
