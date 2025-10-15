# 🏡 Modelo de Predicción de Precios de Viviendas con PyTorch

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

---

### 📖 Resumen del Proyecto

Este proyecto aborda un problema clásico de **regresión**: la predicción del precio de venta de inmuebles. Utilizando un dataset masivo con más de 11 millones de registros, se implementa un flujo de trabajo completo para construir un modelo predictivo basado en una **Red Neuronal Densa (DNN)** con PyTorch.

Debido al gran tamaño del dataset y a la presencia de datos anómalos, el proyecto pone un fuerte énfasis en las etapas de **muestreo, limpieza de datos, y feature engineering** para asegurar la construcción de un modelo robusto.

---

### 🎯 Objetivo

El objetivo es desarrollar un modelo capaz de predecir el precio de una vivienda a partir de sus características (como área, número de habitaciones, ubicación geográfica, etc.), demostrando la capacidad de manejar y modelar datasets a gran escala.

---

### 📊 Dataset

Se utiliza un dataset de anuncios inmobiliarios con **11,358,150 registros** y 15 características.

* **Variable Objetivo:** `price`.
* **Características Principales:** `level` (piso), `rooms` (habitaciones), `area` (superficie), `geo_lat` (latitud), `geo_lon` (longitud), entre otras.

---

### 🛠️ Metodología

1.  **Manejo de Datos a Gran Escala:**
    * Dado que el dataset completo excede la memoria de un entorno de trabajo estándar, se aplicó una estrategia de **muestreo aleatorio** (`frac=0.03`) para crear un subconjunto manejable y representativo para el entrenamiento.

2.  **Limpieza de Datos y Manejo de Outliers:**
    * Se eliminaron columnas con alta cardinalidad y numerosos valores nulos (`postal_code`, `street_id`, etc.).
    * Se filtraron registros con valores inverosímiles (ej. `price < 10000`, `area < 10`).
    * Se trató el problema de los precios extremadamente altos (outliers) eliminando el **0.5% superior** de los datos, basado en el cuantil 99.5.

3.  **Feature Engineering:**
    * Para manejar la distribución sesgada de los precios, se aplicó una **transformación logarítmica** (`np.log1p`) a la variable objetivo. Esto ayuda a que el modelo se entrene de manera más estable.

4.  **Modelado y Entrenamiento:**
    * Se construyó una **Red Neuronal Densa (DNN)** en PyTorch con una arquitectura simple y una capa oculta de 5 neuronas (activación ReLU).
    * Los datos de entrada fueron **estandarizados** usando `StandardScaler`.
    * El modelo fue entrenado utilizando el optimizador `Adam` y la función de pérdida `MSELoss` (Error Cuadrático Medio) durante 5,000 iteraciones.

---

### 📈 Resultados

El modelo entrenado demostró una capacidad clara para aprender la relación entre las características de una vivienda y su precio.

* La **curva de aprendizaje** muestra una disminución consistente del error, indicando una convergencia exitosa.
* El **gráfico de dispersión de valores reales vs. predichos** muestra una tendencia lineal positiva, confirmando que las predicciones del modelo se alinean con los precios reales, aunque con un margen de error que podría mejorarse con arquitecturas más complejas o más datos.
