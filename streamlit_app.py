import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("Heart Regression App - Streamlit")

# Upload dataset
uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data:")
    st.dataframe(df.head())

    # Pilih target numerik saja
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    target_col = st.selectbox("Pilih target (HARUS numerik):", numeric_cols)

    # Fitur
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Mapping kategori angka
    if "Sex" in X:
        X["Sex"] = X["Sex"].map({"M": 1, "F": 0})

    if "ExerciseAngina" in X:
        X["ExerciseAngina"] = X["ExerciseAngina"].map({"Y": 1, "N": 0})

    # Bersihkan text kolom lain
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = (
                X[col]
                .astype(str)
                .str.replace(r"[^A-Za-z0-9]", "", regex=True)
                .str.upper()
            )

    # Onehot encoding dan paksa float
    X_encoded = pd.get_dummies(X, drop_first=True).astype(float)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write("### Hasil Evaluasi Model")
    st.write(f"MSE: {mse:.4f}")
    st.write(f"R² Score: {r2:.4f}")

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Nilai Aktual")
    ax.set_ylabel("Prediksi")
    ax.set_title("Prediksi vs Aktual")
    st.pyplot(fig)

    # Input Manual untuk Prediksi
    st.write("### Coba Prediksi Manual")

    user_input = {}

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            user_input[col] = st.number_input(
                f"Nilai {col}", float(X[col].min()), float(X[col].max())
            )
        else:
            user_input[col] = st.text_input(f"{col} (kategori)").upper().strip()

    if st.button("Prediksi"):
        input_df = pd.DataFrame([user_input])

        # Mapping kategori yang sama seperti training
        if "Sex" in input_df:
            input_df["Sex"] = input_df["Sex"].map({"M": 1, "F": 0})
            input_df["Sex"] = input_df["Sex"].fillna(0)

        if "ExerciseAngina" in input_df:
            input_df["ExerciseAngina"] = input_df["ExerciseAngina"].map({"Y": 1, "N": 0})
            input_df["ExerciseAngina"] = input_df["ExerciseAngina"].fillna(0)

        # Clean categorical text input
        for c in input_df.columns:
            if input_df[c].dtype == object:
                input_df[c] = (
                    input_df[c]
                    .astype(str)
                    .str.replace(r"[^A-Za-z0-9]", "", regex=True)
                    .str.upper()
                )

        # Onehot encoding input
        input_encoded = pd.get_dummies(input_df).astype(float)

        # Samakan kolom
        input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

        # FIX FINAL: Hapus semua NaN 
        input_encoded = input_encoded.fillna(0)

        # Prediksi
        pred = model.predict(input_encoded)
        st.success(f"Prediksi: {pred[0]:,.2f}")

else:
    st.info("Upload file CSV terlebih dahulu.")
