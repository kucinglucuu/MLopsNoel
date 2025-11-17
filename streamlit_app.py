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

    # Pilih kolom target (Y)
    st.write("### Pilih Kolom Target (Y)")
    target_col = st.selectbox("Pilih target:", df.columns)

    # Fitur = semua kolom kecuali target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)
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

    # Prediksi manual
    st.write("### Coba Prediksi Manual")
    input_data = []

    for col in X.columns:
        # Cek apakah kolom numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            val = st.number_input(
                f"Masukkan nilai untuk {col}",
                float(df[col].min()),
                float(df[col].max())
            )
        else:
            # Jika kolom kategorikal -> text input
            val = st.text_input(f"Masukkan nilai untuk {col}")
        input_data.append(val)

    if st.button("Prediksi"):
        # Konversi input ke dataframe agar aman
        input_df = pd.DataFrame([input_data], columns=X.columns)

        # Konversi categorical ke numeric (jika ada)
        input_df = pd.get_dummies(input_df)
        X_model = pd.get_dummies(X)

        # Samakan kolom
        input_df = input_df.reindex(columns=X_model.columns, fill_value=0)

        pred = model.predict(input_df)
        st.success(f"Prediksi nilai: {pred[0]:,.2f}")

else:
    st.info("Silakan upload file CSV terlebih dahulu.")
