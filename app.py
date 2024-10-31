import streamlit as st
import joblib
import numpy as np

# Cargar el modelo
model = joblib.load('modelo_clasificacion.pkl')

# Título de la aplicación
st.title("Predicción de Clasificación Simple")

# Instrucciones
st.write("Introduce los valores de las características para realizar una predicción:")

# Entrada de datos
input1 = st.number_input("Introduce el valor de la primera característica", value=0.0)
input2 = st.number_input("Introduce el valor de la segunda característica", value=0.0)

# Botón para realizar la predicción
if st.button("Realizar Predicción"):
    # Preparar los datos en el formato esperado por el modelo
    entrada = np.array([[input1, input2]])
    # Realizar la predicción
    prediccion = model.predict(entrada)
    # Mostrar el resultado
    st.write("La predicción es:", "Clase 1" if prediccion[0] == 1 else "Clase 0")
