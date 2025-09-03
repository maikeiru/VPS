import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class DodoML:
    def __init__(self, model_path="dodo_ml_model.pkl", vectorizer_path="dodo_ml_vectorizer.pkl"):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self._try_load()

    def _try_load(self):
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(self.vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
        except Exception:
            self.model = None
            self.vectorizer = None

    def entrenar(self, X_train, y_train):
        self.vectorizer = TfidfVectorizer(max_features=600)
        X_vect = self.vectorizer.fit_transform(X_train)
        self.model = LogisticRegression()
        self.model.fit(X_vect, y_train)

    def guardar_modelo(self):
        if self.model and self.vectorizer:
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)
            with open(self.vectorizer_path, "wb") as f:
                pickle.dump(self.vectorizer, f)

    def modelo_cargado(self):
        return self.model is not None and self.vectorizer is not None

    def predecir(self, X):
        if not self.modelo_cargado():
            return [1 for _ in X]  # Assume útil if not trained
        X_vect = self.vectorizer.transform(X)
        return self.model.predict(X_vect)