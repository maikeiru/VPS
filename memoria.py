import os
import json
import sqlite3
from datetime import datetime

class MemoriaDodo:
    """
    Maneja la memoria persistente para DODONEST.
    Permite guardar y recuperar conocimientos, conversaciones, archivos y errores.
    Soporta almacenamiento en SQLite (por defecto) y retrocompatibilidad con JSON.
    """

    def __init__(self, data_dir=".", db_file="dodonest_memoria.db", json_file="dodonest_memoria.json"):
        self.db_path = os.path.join(data_dir, db_file)
        self.json_path = os.path.join(data_dir, json_file)
        os.makedirs(data_dir, exist_ok=True)
        self.use_sqlite = self.init_sqlite()

    def init_sqlite(self):
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute("""CREATE TABLE IF NOT EXISTS conocimientos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                texto TEXT,
                fuente TEXT,
                timestamp TEXT
            )""")
            cursor.execute("""CREATE TABLE IF NOT EXISTS conversaciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pregunta TEXT,
                respuesta TEXT,
                tokens INTEGER,
                api TEXT,
                error TEXT,
                timestamp TEXT
            )""")
            cursor.execute("""CREATE TABLE IF NOT EXISTS archivos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                archivo TEXT,
                analisis TEXT,
                timestamp TEXT
            )""")
            cursor.execute("""CREATE TABLE IF NOT EXISTS errores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error TEXT,
                correccion TEXT,
                timestamp TEXT
            )""")
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error iniciando SQLite: {e}")
            return False

    def guardar_conocimiento(self, texto, fuente):
        ts = datetime.now().isoformat()
        if self.use_sqlite:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO conocimientos (texto, fuente, timestamp) VALUES (?, ?, ?)", (texto, fuente, ts))
            conn.commit()
            conn.close()
        else:
            estado = self._leer_json()
            estado['conocimientos'].append({'texto': texto, 'fuente': fuente, 'timestamp': ts})
            self._guardar_json(estado)

    def guardar_conversacion(self, pregunta, respuesta, tokens, api, error=None):
        ts = datetime.now().isoformat()
        if self.use_sqlite:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO conversaciones (pregunta, respuesta, tokens, api, error, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (pregunta, respuesta, tokens, api, str(error) if error else None, ts))
            conn.commit()
            conn.close()
        else:
            estado = self._leer_json()
            estado['conversaciones'].append({'pregunta': pregunta, 'respuesta': respuesta, 'tokens': tokens, 'api': api, 'error': str(error) if error else None, 'timestamp': ts})
            self._guardar_json(estado)

    def guardar_archivo(self, archivo, analisis):
        ts = datetime.now().isoformat()
        if self.use_sqlite:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO archivos (archivo, analisis, timestamp) VALUES (?, ?, ?)", (archivo, analisis, ts))
            conn.commit()
            conn.close()
        else:
            estado = self._leer_json()
            estado['archivos'].append({'archivo': archivo, 'analisis': analisis, 'timestamp': ts})
            self._guardar_json(estado)

    def guardar_error(self, error, correccion):
        ts = datetime.now().isoformat()
        if self.use_sqlite:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO errores (error, correccion, timestamp) VALUES (?, ?, ?)", (error, correccion, ts))
            conn.commit()
            conn.close()
        else:
            estado = self._leer_json()
            estado.setdefault('errores', []).append({'error': error, 'correccion': correccion, 'timestamp': ts})
            self._guardar_json(estado)

    def get_conocimientos(self, limite=50):
        if self.use_sqlite:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute("SELECT texto FROM conocimientos ORDER BY id DESC LIMIT ?", (limite,))
            resultados = [r[0] for r in cursor.fetchall() if r[0] is not None]
            conn.close()
            return resultados
        else:
            estado = self._leer_json()
            return [c['texto'] for c in estado.get('conocimientos', []) if c.get('texto')][-limite:]

    def get_stats(self):
        if self.use_sqlite:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), SUM(tokens) FROM conversaciones")
            convs, tokens = cursor.fetchone()
            cursor.execute("SELECT COUNT(*) FROM errores")
            errores = cursor.fetchone()[0]
            cursor.execute("SELECT error FROM errores ORDER BY id DESC LIMIT 1")
            ultimo_error = cursor.fetchone()
            conn.close()
            return {
                'total_conversaciones': convs or 0,
                'total_tokens': tokens or 0,
                'errores_api': errores or 0,
                'ultimo_error': ultimo_error[0] if ultimo_error else None
            }
        else:
            estado = self._leer_json()
            return estado.get('stats', {})

    def _leer_json(self):
        if not os.path.exists(self.json_path):
            estado = {'conocimientos': [], 'conversaciones': [], 'archivos': [], 'stats': {}, 'errores': []}
            self._guardar_json(estado)
        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _guardar_json(self, estado):
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(estado, f, indent=2, ensure_ascii=False)