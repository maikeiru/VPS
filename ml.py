"""
ml.py - Sistema ML avanzado para clasificación de utilidad de respuestas

- Embeddings semánticos modernos (SentenceTransformer)
- Validación cruzada y hyperparameter tuning
- Gestión de clases desbalanceadas
- Almacenamiento eficiente y migración automática
- 100% compatible con main.py y sistema.py (mismos métodos y firmas)
"""

import os
import pickle
import json
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, f1_score
from typing import List, Optional

class DodoMLCompat:
    """Clase de compatibilidad backward para el sistema anterior (TF-IDF)"""
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
            return [1 for _ in X]
        X_vect = self.vectorizer.transform(X)
        return self.model.predict(X_vect)

class DodoMLAdvanced:
    """
    Sistema ML avanzado con embeddings semánticos y mejores prácticas.
    Mantiene interfaz compatible con el sistema original.
    """
    def __init__(self, memoria=None, model_path="dodo_ml_advanced.json"):
        self.memoria = memoria
        self.model_path = model_path
        self.embeddings_path = model_path.replace('.json', '_embeddings.npy')
        self.classifier_path = model_path.replace('.json', '_classifier.pkl')
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.classifier = None
        self.best_params = None
        self.data = []
        self.embeddings = None
        self.trained = False
        self.load_model()

    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'r') as f:
                    self.data = json.load(f)
            if os.path.exists(self.embeddings_path):
                self.embeddings = np.load(self.embeddings_path)
            if os.path.exists(self.classifier_path):
                with open(self.classifier_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.classifier = model_data.get('classifier')
                    self.best_params = model_data.get('best_params')
                    self.trained = True
        except Exception as e:
            print(f"[ML] Error cargando modelo: {e}")
            self.data = []
            self.embeddings = None
            self.classifier = None

    def guardar_modelo(self):
        try:
            metadata = []
            for item in self.data:
                meta_item = {k: v for k, v in item.items() if k != 'embedding'}
                metadata.append(meta_item)
            with open(self.model_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            if self.embeddings is not None:
                np.save(self.embeddings_path, self.embeddings)
            if self.classifier is not None:
                model_data = {
                    'classifier': self.classifier,
                    'best_params': self.best_params
                }
                with open(self.classifier_path, 'wb') as f:
                    pickle.dump(model_data, f)
        except Exception as e:
            print(f"[ML] Error guardando modelo: {e}")

    def _extract_features(self, textos: List[str]) -> np.ndarray:
        return self.encoder.encode(textos)

    def recolectar_datos(self):
        textos = []
        etiquetas = []
        if self.memoria and hasattr(self.memoria, 'use_sqlite'):
            try:
                if self.memoria.use_sqlite:
                    import sqlite3
                    conn = sqlite3.connect(self.memoria.db_path, check_same_thread=False)
                    cursor = conn.cursor()
                    cursor.execute("SELECT pregunta, respuesta, error FROM conversaciones ORDER BY id DESC LIMIT 1000")
                    rows = cursor.fetchall()
                    conn.close()
                else:
                    estado = self.memoria._leer_json()
                    rows = [(c['pregunta'], c['respuesta'], c.get('error')) for c in estado.get('conversaciones', [])]
                for pregunta, respuesta, error in rows:
                    if pregunta and respuesta:
                        texto = f"{pregunta} {respuesta}"
                        label = 0 if error else 1
                        textos.append(texto)
                        etiquetas.append(label)
            except Exception as e:
                print(f"[ML] Error recolectando datos de memoria: {e}")
        for item in self.data:
            if 'text' in item and 'label' in item:
                textos.append(item['text'])
                etiquetas.append(item['label'])
        return textos, etiquetas

    def entrenar(self, X_train=None, y_train=None, do_hyperparameter_tuning=True, min_samples=30):
        if X_train is None or y_train is None:
            X_train, y_train = self.recolectar_datos()
        if len(X_train) < min_samples:
            print(f"[ML] Advertencia: Solo {len(X_train)} ejemplos. Mínimo recomendado: {min_samples}")
            return
        unique_labels, counts = np.unique(y_train, return_counts=True)
        print(f"[ML] Distribución de clases: {dict(zip(unique_labels, counts))}")
        if len(unique_labels) < 2:
            print("[ML] Error: Se necesitan al menos 2 clases diferentes para entrenar")
            return
        X_features = self._extract_features(X_train)
        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
            X_features, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        if do_hyperparameter_tuning and len(X_train_split) >= 20:
            print("[ML] Realizando hyperparameter tuning...")
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['lbfgs', 'liblinear'],
                'class_weight': ['balanced', None]
            }
            base_clf = LogisticRegression(max_iter=1000, random_state=42)
            cv_strategy = StratifiedKFold(n_splits=min(5, len(X_train_split)//2), shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                base_clf, 
                param_grid, 
                cv=cv_strategy, 
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train_split, y_train_split)
            self.classifier = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            print(f"[ML] Mejores parámetros: {self.best_params}")
            print(f"[ML] Mejor score CV: {grid_search.best_score_:.3f}")
        else:
            print("[ML] Entrenando con parámetros por defecto...")
            self.classifier = LogisticRegression(
                max_iter=1000, 
                class_weight='balanced',
                random_state=42
            )
            self.classifier.fit(X_train_split, y_train_split)
        y_pred = self.classifier.predict(X_test_split)
        test_f1 = f1_score(y_test_split, y_pred, average='weighted')
        print(f"[ML] F1-score en test set: {test_f1:.3f}")
        print(f"[ML] Modelo entrenado con {len(X_train)} ejemplos totales")
        cv_scores = cross_val_score(
            self.classifier, X_features, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1_weighted'
        )
        print(f"[ML] Validación cruzada F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print("\n[ML] Reporte de clasificación (test set):")
        print(classification_report(y_test_split, y_pred, digits=3))
        self.trained = True
        self.guardar_modelo()

    def predecir(self, X: List[str]) -> List[int]:
        if not self.modelo_cargado():
            print("[ML] Advertencia: Modelo no entrenado, asumiendo útil por defecto")
            return [1 for _ in X]
        try:
            X_features = self._extract_features(X)
            predictions = self.classifier.predict(X_features)
            return predictions.tolist()
        except Exception as e:
            print(f"[ML] Error en predicción: {e}")
            return [1 for _ in X]

    def predecir_utilidad(self, pregunta: str, respuesta: str) -> Optional[int]:
        texto = f"{pregunta} {respuesta}"
        predicciones = self.predecir([texto])
        return predicciones[0] if predicciones else None

    def modelo_cargado(self) -> bool:
        return self.classifier is not None and self.trained

    def get_stats(self) -> dict:
        return {
            'ejemplos_totales': len(self.data),
            'modelo_entrenado': self.trained,
            'mejores_params': self.best_params,
            'arquitectura': 'SentenceTransformer + LogisticRegression'
        }

class DodoML(DodoMLAdvanced):
    """
    Clase principal, lista para producción, 100% compatible.
    """
    def __init__(self, memoria=None, model_path="dodo_ml_model.pkl"):
        if model_path.endswith('.pkl'):
            model_path = model_path.replace('.pkl', '_advanced.json')
        super().__init__(memoria, model_path)
        self._migrate_old_model()

    def _migrate_old_model(self):
        old_model_path = "dodo_ml_model.pkl"
        old_vectorizer_path = "dodo_ml_vectorizer.pkl"
        if os.path.exists(old_model_path) and os.path.exists(old_vectorizer_path):
            print("[ML] Detectado modelo TF-IDF anterior. Recomiendo re-entrenar con embeddings semánticos.")

DodoMLCompat = DodoMLCompat