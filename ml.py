import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class DodoML:
    """
    Clasificador simple de utilidad de respuesta y errores.
    Aprende de las conversaciones y errores guardados en la memoria.
    """

    def __init__(self, memoria, model_path="dodo_ml_model.pkl"):
        self.memoria = memoria
        self.model_path = model_path
        self.vectorizer = TfidfVectorizer(max_features=200)
        self.model = LogisticRegression(class_weight='balanced')
        self.trained = False
        self.load_model()

    def recolectar_datos(self):
        """
        Extrae preguntas, respuestas y etiquetas de utilidad/error de la memoria.
        Etiqueta: 1 si la respuesta NO tiene error, 0 si tiene error.
        """
        conversaciones = []
        etiquetas = []
        if self.memoria.use_sqlite:
            import sqlite3
            conn = sqlite3.connect(self.memoria.db_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute("SELECT pregunta, respuesta, error FROM conversaciones ORDER BY id DESC LIMIT 500")
            rows = cursor.fetchall()
            conn.close()
        else:
            estado = self.memoria._leer_json()
            rows = [(c['pregunta'], c['respuesta'], c.get('error')) for c in estado.get('conversaciones', [])]
        for pregunta, respuesta, error in rows:
            texto = f"{pregunta} {respuesta}"
            label = 0 if error else 1
            conversaciones.append(texto)
            etiquetas.append(label)
        return conversaciones, etiquetas

    def entrenar(self):
        """
        Entrena el clasificador con los datos de la memoria.
        Ahora muestra un reporte detallado antes de entrenar.
        """
        X, y = self.recolectar_datos()
        total = len(X)
        utiles = sum(1 for label in y if label == 1)
        errores = sum(1 for label in y if label == 0)
        clases = set(y)
        print(f"[ML] Ejemplos totales: {total} | Útiles: {utiles} | Errores: {errores} | Clases presentes: {clases}")

        if total < 10:
            print("No hay suficientes datos para entrenar el modelo ML (mínimo 10).")
            return
        if len(clases) < 2:
            print("No hay suficientes clases distintas para entrenar el modelo ML (se necesita al menos 1 útil y 1 error).")
            return
        if utiles < 3 or errores < 3:
            print("No hay suficiente representación de ambas clases (mínimo 3 por clase recomendado).")
            return

        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)
        self.trained = True
        self.save_model()
        print(f"Modelo ML entrenado con {total} ejemplos (útiles: {utiles}, errores: {errores}).")

    def predecir_utilidad(self, pregunta, respuesta):
        """
        Predice si una respuesta será útil (sin error) o no.
        Retorna 1 (útil) o 0 (error probable).
        """
        if not self.trained:
            return None
        texto = f"{pregunta} {respuesta}"
        X_vec = self.vectorizer.transform([texto])
        pred = self.model.predict(X_vec)[0]
        return pred

    def save_model(self):
        """
        Guarda el modelo entrenado en disco.
        """
        with open(self.model_path, "wb") as f:
            pickle.dump((self.vectorizer, self.model), f)

    def load_model(self):
        """
        Carga el modelo entrenado si existe.
        """
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.vectorizer, self.model = pickle.load(f)
            self.trained = True