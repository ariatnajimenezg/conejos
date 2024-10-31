from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import joblib

# Generar datos de ejemplo
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Entrenar el modelo
model = RandomForestClassifier()
model.fit(X, y)

# Guardar el modelo en un archivo
joblib.dump(model, 'modelo_clasificacion.pkl')
print("Modelo entrenado y guardado como modelo_clasificacion.pkl")
