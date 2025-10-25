import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# Importamos la implementación de regresión lineal de Scikit-learn
from sklearn.linear_model import LinearRegression


class LinearRegressionModel:
    """Clase de modelo de Regresión Lineal usando Scikit-learn."""

    def __init__(self):
        # Usamos el modelo robusto de sklearn
        self.model = LinearRegression()

    def train(self, X, y):
        # El método fit de sklearn maneja las dimensiones 2D de X
        self.model.fit(X, y)

    def predict(self, X):
        # El método predict de sklearn devuelve las predicciones
        return self.model.predict(X)

    def score(self, X, y):
        """Calcula el coeficiente R^2 usando el método incorporado de sklearn."""
        # El método score de sklearn devuelve directamente el R^2
        return self.model.score(X, y)

    def plot_regression(self, X, y, feature_name, target_name):
        """Genera y devuelve la figura de Matplotlib."""
        # Aseguramos que X sea 1D para los ejes de la gráfica
        X_flat = X.flatten()

        fig, ax = plt.subplots(figsize=(10, 6))

        # 1. Graficar los puntos de datos (Scatter Plot)
        ax.scatter(X_flat, y, color="blue", label="Datos reales", alpha=0.6)

        # 2. Generar predicciones para la línea
        # Creamos un rango de X para dibujar la línea suave
        # Usamos X_flat para calcular los límites
        X_min = X_flat.min()
        X_max = X_flat.max()
        X_fit = np.linspace(X_min * 0.9, X_max * 1.1, 100).reshape(-1, 1)
        y_fit = self.predict(X_fit)

        # Obtenemos los coeficientes para la leyenda
        slope = self.model.coef_[0]
        intercept = self.model.intercept_

        # 3. Graficar la línea de regresión
        ax.plot(
            X_fit.flatten(),
            y_fit,
            color="red",
            linewidth=2,
            label=f"Línea de Regresión (m={slope:.2f}, b={intercept:.2f})",
        )

        # Configurar la gráfica
        ax.set_title(f"Regresión Lineal: {feature_name} vs {target_name}")
        ax.set_xlabel(feature_name)
        ax.set_ylabel(target_name)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

        # Devolver la figura
        return fig


def load_data(uploaded_file):

    # Intenta cargar el archivo usando diferentes codificaciones
    encodings_to_try = ["utf-8", "latin-1", "cp1252"]
    data = None

    for encoding in encodings_to_try:
        try:
            # Intentar primero con el delimitador estándar (,)
            uploaded_file.seek(0)
            data = pd.read_csv(uploaded_file, encoding=encoding)

            # Verificar si se necesita el delimitador punto y coma (;)
            if len(data.columns) <= 1 and encoding != "cp1252":
                # Si solo hay una columna, probamos con punto y coma (;)
                uploaded_file.seek(0)
                data = pd.read_csv(uploaded_file, encoding=encoding, sep=";")

            if len(data.columns) > 1:
                break  # Éxito en la carga
        except Exception:
            data = None

    if data is None:
        st.error(
            "Error al cargar el archivo. Asegúrate de que el formato sea CSV (separado por comas o punto y coma) y la codificación sea compatible."
        )
        return pd.DataFrame()  # Devolver un DataFrame vacío

    # Limpieza y conversión a numérico
    for col in data.columns:
        # Convertir a numérico, forzando errores a NaN
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Eliminar filas con NaN que resultaron de la conversión forzada
    data.dropna(inplace=True)
    return data


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

    if data.empty:
        st.warning("No hay datos numéricos válidos para el análisis.")
    else:
        st.subheader("Datos Cargados")

        # Vista previa de datos cargados
        st.write("Vista previa de los datos:")
        st.dataframe(data.head(10))

        # ------------------------------------------------------------------
        ## 2. Selección de Variables
        # ------------------------------------------------------------------
        col_list = data.columns.tolist()
        if len(col_list) < 2:
            st.error(
                "Se necesitan al menos dos columnas numéricas para realizar la regresión."
            )
        else:
            col1, col2 = st.columns(2)

            with col1:
                feature_column = st.selectbox(
                    "Selecciona la variable independiente (X)",
                    col_list,
                    index=0,
                    key="feature_select",
                )
            with col2:
                # Asegura que la variable Y no sea la misma que X por defecto
                default_target_index = (
                    1 if len(col_list) > 1 and feature_column == col_list[0] else 0
                )
                target_column = st.selectbox(
                    "Selecciona la variable dependiente (Y)",
                    col_list,
                    index=default_target_index,
                    key="target_select",
                )

            # ------------------------------------------------------------------
            ## 3. Entrenamiento del Modelo
            # ------------------------------------------------------------------
            with st.form("form_entrenamiento"):
                st.subheader("Entrenamiento del Modelo")

                submit_train = st.form_submit_button(
                    "Entrenar modelo y Mostrar Resultados"
                )

                if submit_train and feature_column != target_column:
                    # Reestructurar los datos para el modelo (X debe ser 2D)
                    # **ESTO ES CLAVE: ASEGURA LA FORMA (N, 1)**
                    X = data[feature_column].values.reshape(-1, 1)
                    y = data[target_column].values

                    model = LinearRegressionModel()
                    model.train(X, y)

                    # Guardar el modelo (Asegúrate de que el modelo sea el objeto de sklearn)
                    with open("modelo_regresion.pkl", "wb") as file:
                        # Guardamos el objeto self.model (que es la instancia de sklearn)
                        pickle.dump(model.model, file)

                    # Llama al método score()
                    r_squared = model.score(X, y)

                    st.success("¡Modelo entrenado exitosamente!")

                    # Muestra los coeficientes para el debugging visual
                    st.info(f"**R² (Coeficiente de Determinación):** {r_squared:.4f}")
                    st.write(f"**Pendiente (m):** {model.model.coef_[0]:.4f}")
                    st.write(f"**Intercepto (b):** {model.model.intercept_:.4f}")

                    # Muestra la gráfica
                    fig = model.plot_regression(X, y, feature_column, target_column)
                    st.pyplot(fig)

                elif submit_train and feature_column == target_column:
                    st.error(
                        "Las variables independiente y dependiente no pueden ser la misma."
                    )

            # ------------------------------------------------------------------
            ## 4. Predicciones
            # ------------------------------------------------------------------
            if os.path.exists("modelo_regresion.pkl"):
                with st.form("form_prediccion"):
                    st.subheader("Hacer Predicciones")

                    # Usamos los valores min/max de la columna X para el rango de entrada
                    min_val = data[feature_column].min()
                    max_val = data[feature_column].max()

                    input_value = st.number_input(
                        f"Ingresa un valor para **{feature_column}** (Rango: {min_val:.2f} a {max_val:.2f})",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(np.mean(data[feature_column])),
                        key="input_predict",
                    )

                    submit_predict = st.form_submit_button("Predecir")

                    if submit_predict:
                        with open("modelo_regresion.pkl", "rb") as file:
                            loaded_model = pickle.load(file)

                        # La predicción espera un array 2D
                        prediction = loaded_model.predict(np.array([[input_value]]))

                        st.metric(
                            f"Predicción de {target_column}", f"{prediction[0]:.2f}"
                        )
