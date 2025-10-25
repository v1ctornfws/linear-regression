from sklearn.linear_model import LinearRegression
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np # Necesario para algunas funciones de la gráfica

class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    # ----------------------------------------------------
    # NUEVO: Método para obtener el R^2 (Coeficiente de Determinación)
    def score(self, X, y):
        """Devuelve el coeficiente R^2 del modelo."""
        return self.model.score(X, y)

    def plot_regression(self, X, y, x_label, y_label):
        # Creamos un objeto figura y ejes
        fig, ax = plt.subplots(figsize=(10, 6))

        # 1. Scatter de los datos
        ax.scatter(X, y, color="blue", alpha=0.5, label='Datos reales')

        # 2. Preparamos los datos para la línea de regresión (ordenamos X para dibujar la línea)
        X_fit = np.sort(X, axis=0)
        y_fit = self.model.predict(X_fit)
        
        # 3. Dibujamos la línea de regresión
        ax.plot(X_fit, y_fit, color="red", linewidth=2, label='Línea de Regresión')
        
        # Configuración de etiquetas y título
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title("Regresión Lineal")
        ax.legend()
        ax.grid(True)
        
        # En lugar de st.pyplot(plt), devolvemos la figura para que app.py la maneje
        return fig