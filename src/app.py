import streamlit as st
import pandas as pd
import numpy as np
from models.linear_regression import LinearRegressionModel
from utils.data_loader import load_data
import pickle
import os

# Configuración de la página
st.set_page_config(
    page_title="Aplicaciones Web - Regresion lineal", page_icon="🧮", layout="wide"
)

st.title("Aplicación de Regresión Lineal")

# Subida de archivo
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("Vista previa de los datos:")
    st.write(data.head(10))

    # Selección de variables
    feature_column = st.selectbox(
        "Selecciona la variable independiente (X)", data.columns
    )
    target_column = st.selectbox("Selecciona la variable dependiente (Y)", data.columns)

    if st.button("Entrenar modelo"):
        X = data[feature_column].values.reshape(-1, 1)
        y = data[target_column].values

        model = LinearRegressionModel()
        model.train(X, y)

        # Guardar el modelo
        with open("modelo_regresion.pkl", "wb") as file:
            pickle.dump(model, file)

        st.success("¡Modelo entrenado exitosamente!")

        # Mostrar gráfica
        model.plot_regression(X, y, feature_column, target_column)

    # Predicción
    if os.path.exists("modelo_regresion.pkl"):
        st.subheader("Hacer predicciones")
        input_value = st.number_input("Ingresa un valor para predecir")

        if st.button("Predecir"):
            with open("modelo_regresion.pkl", "rb") as file:
                loaded_model = pickle.load(file)
            prediction = loaded_model.predict([[input_value]])
            st.write(f"Predicción: {prediction[0]:.2f}")

# ... código anterior ...
    feature_column = st.selectbox(
        "Selecciona la variable independiente (X)", data.columns
    )
    target_column = st.selectbox("Selecciona la variable dependiente (Y)", data.columns)

# -------------------------------------------------------------
# NUEVA ESTRUCTURA con st.form para el entrenamiento
# -------------------------------------------------------------
    with st.form("form_entrenamiento"):
        st.subheader("Entrenamiento del Modelo")
        
        # El botón de entrenamiento ahora está DENTRO del formulario
        submit_train = st.form_submit_button("Entrenar modelo")

        if submit_train:
            X = data[feature_column].values.reshape(-1, 1)
            y = data[target_column].values

            model = LinearRegressionModel()
            model.train(X, y)

            # Guardar el modelo
            with open("modelo_regresion.pkl", "wb") as file:
                pickle.dump(model, file)

            # Calcular R^2
            r_squared = model.score(X, y) 

            st.success("¡Modelo entrenado exitosamente!")
            st.info(f"R² (Coeficiente de Determinación): {r_squared:.4f}")

            # Mostrar gráfica (asumiendo que devuelve una figura)
            fig = model.plot_regression(X, y, feature_column, target_column)
            st.pyplot(fig)

    if os.path.exists("modelo_regresion.pkl"):
        with st.form("form_prediccion"):
            st.subheader("Hacer predicciones")
            input_value = st.number_input("Ingresa un valor para predecir")
            
            # El botón de predicción ahora está DENTRO del formulario
            submit_predict = st.form_submit_button("Predecir")
            
            if submit_predict:
                with open("modelo_regresion.pkl", "rb") as file:
                    loaded_model = pickle.load(file)
                
                prediction = loaded_model.predict([[input_value]])
                st.write(f"Predicción: {prediction[0]:.2f}")