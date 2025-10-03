🌸 Iris-KNN

Este proyecto implementa un clasificador de vecinos más cercanos (K-Nearest Neighbors, KNN) para predecir la especie de flores del conjunto de datos clásico Iris. Utiliza scikit-learn para el modelado y kagglehub para descargar automáticamente el dataset desde Kaggle.
📦 Requisitos

    Python 3.8+

    pip

📚 Dependencias

Instala las dependencias necesarias con:
    
    pip install pandas numpy scikit-learn kagglehub

🚀 Ejecución
  
    import kagglehub
    path = kagglehub.dataset_download("uciml/iris")
  1. Descarga el dataset Iris desde Kaggle.
  2. Limpia y codifica los datos.
  3. Divide el conjunto en entrenamiento y prueba.
  4. Escala las características.
  5. Entrena un modelo KNN con n_neighbors=3.
  6. Evalúa el modelo con métricas de precisión, matriz de confusión y reporte de clasificación.
  7. Guarda el modelo y el escalador en archivos .pkl.

📊 Resultados
El modelo genera:
  - Precisión (accuracy_score)
  - Reporte de clasificación (classification_report)
  - Matriz de confusión (confusion_matrix)

Estos resultados permiten evaluar el rendimiento del clasificador sobre el conjunto de prueba.

💾 Archivos generados
  - knn_model.pkl: modelo KNN entrenado
  - knn_scaler.pkl: objeto StandardScaler usado para escalar los datos

📁 Estructura del proyecto
  
    iris-knn/
    ├── knn_model.pkl
    ├── knn_scaler.pkl
    ├── main.py
    └── README.md
