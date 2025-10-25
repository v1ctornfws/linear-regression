import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt  # Necesario para st.pyplot()

# Importar tus módulos (asegúrate de que existan)
# from models.linear_regression import LinearRegressionModel
# from utils.data_loader import load_data

# *** ASUNCIONES TEMPORALES PARA QUE EL CÓDIGO CORRA COMPLETO ***
# Si estás subiendo un archivo, DEBES tener estas funciones
# Las defino aquí con implementaciones básicas si no tienes tus archivos listos:


class LinearRegressionModel:
    """Clase de modelo de Regresión Lineal con implementacion basica para este demo."""

    def __init__(self):
        self.slope = 0.0
        self.intercept = 0.0

    def train(self, X, y):
        # Implementación simple de mínimos cuadrados
        X_mean = np.mean(X)
        y_mean = np.mean(y)

        numerator = np.sum((X - X_mean) * (y - y_mean))
        denominator = np.sum((X - X_mean) ** 2)

        # Evitar división por cero
        if denominator != 0:
            self.slope = numerator / denominator
        else:
            self.slope = 0.0

        self.intercept = y_mean - (self.slope * X_mean)

    def predict(self, X):
        return self.slope * X.flatten() + self.intercept

    def score(self, X, y):
        """Calcula el coeficiente R^2."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        # Evitar división por cero
        if ss_tot == 0:
            return 0.0
        return 1 - (ss_res / ss_tot)

    def plot_regression(self, X, y, feature_name, target_name):
        """Genera y devuelve la figura de Matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # 1. Graficar los puntos de datos (Scatter Plot)
        ax.scatter(X, y, color="blue", label="Datos reales")

        # 2. Generar predicciones para la línea
        # Creamos un rango de X para dibujar la línea suave
        X_fit = np.linspace(X.min() * 0.9, X.max() * 1.1, 100).reshape(-1, 1)
        y_fit = self.predict(X_fit)

        # 3. Graficar la línea de regresión
        ax.plot(
            X_fit,
            y_fit,
            color="red",
            label=f"Línea de Regresión (m={self.slope:.2f}, b={self.intercept:.2f})",
        )

        # Configurar la gráfica
        ax.set_title(f"Regresión Lineal: {feature_name} vs {target_name}")
        ax.set_xlabel(feature_name)
        ax.set_ylabel(target_name)
        ax.legend()
        ax.grid(True)

        # Devolver la figura
        return fig


def load_data(uploaded_file):
    """Carga los datos desde un archivo CSV."""
    # Intentar leer el CSV con coma, luego con punto y coma
    try:
        data = pd.read_csv(uploaded_file, encoding="latin1")
    except:
        uploaded_file.seek(0)  # reset file pointer
        data = pd.read_csv(uploaded_file, sep=";", encoding="latin1")

    # Intenta convertir las columnas a numéricas, forzando errores a NaN
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Eliminar filas con NaN que resultaron de la conversión forzada
    data.dropna(inplace=True)
    return data


# *** FIN DE ASUNCIONES TEMPORALES ***


# Configuración de la página
st.set_page_config(
    page_title="Aplicaciones Web - Regresion lineal", page_icon="🧮", layout="wide"
)

st.title("Aplicación de Regresión Lineal")

# ------------------------------------------------------------------
## 1. Subida y Previsualización de Datos
# ------------------------------------------------------------------
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.subheader("Datos Cargados")

    # Vista previa de datos cargados
    st.write("Vista previa de los datos:")
    st.dataframe(data.head())  # Usamos st.dataframe para una mejor visualización

    # ------------------------------------------------------------------
    ## 2. Selección de Variables (Corregido con 'key')
    # ------------------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        feature_column = st.selectbox(
            "Selecciona la variable independiente (X)",
            data.columns,
            key="feature_select",  # Clave única para evitar StreamlitDuplicateElementId
        )
    with col2:
        target_column = st.selectbox(
            "Selecciona la variable dependiente (Y)",
            data.columns,
            key="target_select",  # Clave única para evitar StreamlitDuplicateElementId
        )

    # ------------------------------------------------------------------
    ## 3. Entrenamiento del Modelo (Corregido con 'st.form')
    # ------------------------------------------------------------------
    with st.form("form_entrenamiento"):
        st.subheader("Entrenamiento del Modelo")

        # Usamos st.form_submit_button para aislar el ID del botón
        submit_train = st.form_submit_button("Entrenar modelo y Mostrar Resultados")

        if submit_train:
            # Reestructurar los datos para el modelo (X debe ser 2D)
            X = data[feature_column].values.reshape(-1, 1)
            y = data[target_column].values

            model = LinearRegressionModel()
            model.train(X, y)

            # Guardar el modelo
            with open("modelo_regresion.pkl", "wb") as file:
                pickle.dump(model, file)

            # Calcular y mostrar R^2
            r_squared = model.score(X, y)
            st.success("✅ ¡Modelo entrenado exitosamente!")

            # Muestra los coeficientes y R^2
            st.info(
                f"**Ecuación de Regresión:** $Y = {model.slope:.2f}X + {model.intercept:.2f}$"
            )
            st.info(f"**R² (Coeficiente de Determinación):** {r_squared:.4f}")

            # Mostrar gráfica con st.pyplot()
            st.subheader("Visualización del Modelo")
            fig = model.plot_regression(X, y, feature_column, target_column)
            st.pyplot(fig)

    # ------------------------------------------------------------------
    ## 4. Predicción (Corregido con 'st.form')
    # ------------------------------------------------------------------
    if os.path.exists("modelo_regresion.pkl"):
        with st.form("form_prediccion"):
            st.subheader("Hacer Predicciones")

            # Añadir una clave al number_input por seguridad
            input_value = st.number_input(
                f"Ingresa un valor para **{feature_column}**", key="input_predict"
            )

            # Botón de predicción dentro de su propio formulario
            submit_predict = st.form_submit_button("Predecir")

            if submit_predict:
                with open("modelo_regresion.pkl", "rb") as file:
                    loaded_model = pickle.load(file)

                # La predicción espera un array 2D
                prediction = loaded_model.predict(np.array([[input_value]]))

                st.metric(f"Predicción de {target_column}", f"{prediction[0]:.2f}")
