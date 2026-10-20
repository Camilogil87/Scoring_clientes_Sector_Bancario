# Scoring_clientes_Sector_Bancario
Proyecto de Machine Learning desarrollado para Azierta Servicios Financieros, orientado a optimizar una campaña de telemarketing de Bankinter Consumer Finance mediante un modelo de scoring de clientes. Su objetivo es estimar la probabilidad de contratación del producto CrediMax y priorizar los listados de llamadas según la propensión de compra.
El proyecto incluye extracción de datos con SQL, anonimización, limpieza y análisis exploratorio con Python, además de ingeniería de variables y tratamiento del desbalance de clases. Se compararon modelos de regresión logística, Random Forest y XGBoost, utilizando MLflow para registrar experimentos y evaluar configuraciones de hiperparámetros con métricas de precisión, recall y F1.

La regresión logística obtuvo el mejor rendimiento para el objetivo planteado. El resultado es una puntuación por cliente que permite orientar la gestión comercial hacia los contactos con mayor probabilidad de conversión.
