import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading, os, json, re, sqlite3, random
from datetime import datetime
import openai
import time
import requests
import difflib

try: import google.generativeai as genai; GEMINI=True
except: GEMINI=False
try: from PIL import Image; import pytesseract; PILLOW=True
except: PILLOW=False
try: import fitz; PDF=True
except: PDF=False
try: import pickle
except: pass
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Check if transformers and GPT2 are available
try:
    _ = GPT2LMHeadModel, GPT2Tokenizer
    GPT2_AVAILABLE = True
except Exception:
    GPT2_AVAILABLE = False

MEMORY_FILE = "gpt2_memory.json"

def save_to_memory(query, answer):
    memory = load_memory()
    memory[query] = answer
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def search_memory(query):
    memory = load_memory()
    return memory.get(query, None)

MODEL_PATH = "gpt2"  # Usar modelo base GPT-2 en inglés, más estable

# Clase GPT2Local CORREGIDA y FUNCIONAL
class GPT2Local:
    def __init__(self, model_path="gpt2"):  # Usar modelo base gpt2 en inglés
        self.available = False
        self.model = None
        self.tokenizer = None
        
        if not GPT2_AVAILABLE:
            print("⚠️ GPT-2 no disponible: Transformers no instalado")
            return
            
        try:
            print("🔄 Cargando modelo GPT-2...")
            # Usar modelo base GPT-2 (más estable que el español)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
            
            # Configurar pad_token si no existe
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model.eval()
            self.available = True
            print("✅ Modelo GPT-2 cargado correctamente")
        except Exception as e:
            print(f"❌ Error cargando GPT-2: {e}")
            self.available = False


    def generate(self, prompt, max_length=100, temperature=0.8, top_p=0.9):
        if not self.available:
            return "Modelo GPT-2 local no disponible."
        try:
            structured_prompt = f"DODONEST AI Assistant created by Maikeiru.\nUser: {prompt}\nDODONEST:"
            inputs = self.tokenizer.encode(structured_prompt, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=min(len(inputs[0]) + 50, max_length),
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "DODONEST:" in generated_text:
                response = generated_text.split("DODONEST:")[-1].strip()
            else:
                response = generated_text[len(structured_prompt):].strip()
            response = self.clean_response(response)
            if len(response.split()) < 3 or len(response) < 10:
                return "No puedo generar respuesta útil con el modelo local."
            return response
        except Exception as e:
            print(f"❌ Error generando con GPT-2 local: {e}")
            return "No puedo generar respuesta útil con el modelo local."

    def clean_response(self, response):
        if not response:
            return ""
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 2:
                if not any(line.lower() in prev.lower() for prev in cleaned_lines[-2:]):
                    cleaned_lines.append(line)
            if len(cleaned_lines) >= 3:
                break
        result = ' '.join(cleaned_lines)
        if len(result) > 200:
            result = result[:200] + "..."
        return result

    def fine_tune(self, train_texts, epochs=2, lr=5e-5):
        if not self.available:
            print("Modelo GPT-2 local no disponible para entrenamiento.")
            return
        from torch.optim import AdamW
        optimizer = AdamW(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            for i, text in enumerate(train_texts):
                try:
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    labels = inputs.input_ids.clone()
                    outputs = self.model(**inputs, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                except Exception as e:
                    print(f"Error en fine-tuning texto {i}: {e}")
                    continue
        self.model.eval()

    def save_model(self, path="./dodonest_model"):
        if not self.available:
            print("Modelo GPT-2 local no disponible para guardar.")
            return
        try:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        except Exception as e:
            print(f"❌ Error guardando modelo: {e}")

    def load_model(self, path="./dodonest_model"):
        try:
            if os.path.exists(path):
                self.model = GPT2LMHeadModel.from_pretrained(path)
                self.tokenizer = GPT2Tokenizer.from_pretrained(path)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.eval()
                self.available = True
            else:
                print(f"⚠️ No se encontró modelo en {path}, usando modelo base")
        except Exception as e:
            print(f"❌ Error cargando modelo personalizado: {e}")
            self.available = False

DB_FILE = "dodonest_memoria.db"
JSON_MEMFILE = "dodonest_memoria.json"
DATA_DIR = r"C:\Users\mikol\OneDrive\Escritorio\DODONEST"
os.makedirs(DATA_DIR, exist_ok=True)

# --- Configuración segura de APIs ---
def load_api_keys():
    config_file = "config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get('OPENAI_API_KEY'), config.get('GEMINI_API_KEY')
        except:
            pass
    OPENAI_API_KEY = "REMOVIDOproj-2wlRQZ-r6tys-B3zsTU_e04PGxdq9SwTP0oVnzWuMsHF2XFnBTV06-xtusGSvpp-bgtSg93RItT3BlbkFJO5OzCdGQWz05C7OetoB6Ld86OhJLp9ZfC5qVM47I_oZj58l3YujZCqFa86_Rg-ugDg5IOCjdsA"
    GEMINI_API_KEY = "AIzaSyDM3BcG2YZtZJMgi6IAlAb19r7t0ouxuDQ"
    return OPENAI_API_KEY, GEMINI_API_KEY

def validate_openai_key(api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        return True, client
    except openai.AuthenticationError:
        return False, None
    except Exception as e:
        print(f"Error validando OpenAI: {e}")
        return False, None

OPENAI_API_KEY, GEMINI_API_KEY = load_api_keys()
openai_valid, openai_client = validate_openai_key(OPENAI_API_KEY)
if not openai_valid:
    print("⚠️ Clave de OpenAI inválida o problema de conectividad")
    openai_client = None

if GEMINI:
    try:  # ✅ AGREGAR DOS PUNTOS
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')
        test = gemini_model.generate_content("Test")
        print("✅ Gemini configurado correctamente")
    except Exception as e:
        print(f"⚠️ Error configurando Gemini: {e}")
        GEMINI = False
        
# --- Memoria avanzada: SQLite + retrocompatibilidad JSON ---
class MemoriaDodo:
    def __init__(self):
        self.db_path = os.path.join(DATA_DIR, DB_FILE)
        self.json_path = os.path.join(DATA_DIR, JSON_MEMFILE)
        self.use_sqlite = self.init_sqlite()

    def init_sqlite(self):
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            # Tablas principales
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

# --- Move SistemaHibridoInteligente to global scope ---
class SistemaHibridoInteligente:
    def __init__(self, sistema_base):
        self.sistema = sistema_base  # Tu DODONEST actual
        self.stats_ahorro = {
            'tokens_ahorrados': 0,
            'usos_gpt2_exitosos': 0,
            'usos_api_completos': 0,
            'mejoras_aplicadas': 0
        }
        
    def evaluar_contenido_gpt2(self, contenido, mensaje):
        """Evalúa si el contenido de GPT-2 es útil como base"""
        if not contenido or len(contenido.strip()) < 20:
            return 0.0, "Contenido muy corto"
            
        # Verificaciones de calidad
        puntuaciones = {
            'longitud_util': 1.0 if len(contenido.split()) > 15 else 0.3,
            'sin_basura': 0.0 if self.contiene_basura(contenido) else 1.0,
            'relevancia': self.calcular_relevancia(contenido, mensaje),
            'coherencia': self.evaluar_coherencia(contenido),
            'informativo': 1.0 if any(palabra in contenido.lower() for palabra in ['porque', 'debido', 'consiste', 'significa', 'ejemplo']) else 0.5
        }
        
        score = sum(puntuaciones.values()) / len(puntuaciones)
        razon = f"Longitud: {puntuaciones['longitud_util']:.1f}, Sin basura: {puntuaciones['sin_basura']:.1f}, Relevancia: {puntuaciones['relevancia']:.1f}"
        
        return score, razon
    
    def contiene_basura(self, contenido):
        """Detecta si el contenido tiene texto basura"""
        basura_patterns = [
            "géro", "fresí", "DAKO", "[RNG]", "want to learn the game",
            "llegar una entre", "abajo con hace", "nuevo donde que"
        ]
        contenido_lower = contenido.lower()
        return any(pattern.lower() in contenido_lower for pattern in basura_patterns)
    
    def calcular_relevancia(self, contenido, mensaje):
        """Calcula qué tan relevante es el contenido al mensaje"""
        palabras_mensaje = set(mensaje.lower().split())
        palabras_contenido = set(contenido.lower().split())
        
        if not palabras_mensaje:
            return 0.0
            
        coincidencias = len(palabras_mensaje.intersection(palabras_contenido))
        relevancia = coincidencias / len(palabras_mensaje)
        
        return min(relevancia * 2, 1.0)  # Multiplicar por 2 pero máximo 1.0
    
    def evaluar_coherencia(self, contenido):
        """Evalúa si el contenido es coherente"""
        oraciones = contenido.split('.')
        oraciones_validas = [o.strip() for o in oraciones if len(o.strip()) > 10]
        
        if len(oraciones_validas) == 0:
            return 0.0
        elif len(oraciones_validas) == 1:
            return 0.7
        else:
            return 1.0
    
    def crear_prompt_hibrido(self, mensaje, contenido_base, score_gpt2):
        """Crea prompt optimizado para que API mejore contenido GPT-2"""
        if score_gpt2 > 0.7:
            # GPT-2 tiene buen contenido, solo necesita pulido
            return f"""Eres DODONEST, una IA avanzada creada por Maikeiru.

Tienes este contenido base útil: "{contenido_base}"

Úsalo como foundation y crea una respuesta elaborada y profesional para: "{mensaje}"

Instrucciones:
- Mantén los hechos correctos del contenido base
- Elabora y añade detalles profesionales
- Estructura la respuesta en párrafos claros
- Añade ejemplos si es apropiado
- Mantén un tono inteligente y útil como DODONEST"""

        elif score_gpt2 > 0.4:
            # GPT-2 tiene algo útil pero necesita corrección
            return f"""Eres DODONEST, una IA avanzada creada por Maikeiru.

Tienes este contenido base parcial: "{contenido_base}"

Úsalo como referencia pero corrígelo según sea necesario para responder: "{mensaje}"

Instrucciones:
- Corrige cualquier error del contenido base
- Añade información faltante importante
- Elabora una respuesta completa y detallada
- Si el contenido base es incorrecto, prioriza la precisión"""

        else:
            # GPT-2 no es útil, API genera desde cero
            return f"""Eres DODONEST, una IA avanzada creada por Maikeiru.

Responde de forma elaborada y detallada: "{mensaje}"

(Nota: El contenido base local no fue útil, genera respuesta completa)"""
    
    def generar_respuesta_hibrida(self, mensaje, motor="hibrido"):
        """Función principal del sistema híbrido"""
        
        # 1. INTENTAR GPT-2 COMO BASE
        print(f"🔄 Generando contenido base con GPT-2...")
        contenido_gpt2 = self.sistema.generar_respuesta_gpt2local(mensaje)
        score_gpt2, razon_score = self.evaluar_contenido_gpt2(contenido_gpt2, mensaje)
        
        print(f"📊 GPT-2 Score: {score_gpt2:.2f} - {razon_score}")
        
        # 2. DECIDIR ESTRATEGIA
        if score_gpt2 > 0.4:  # GPT-2 tiene algo útil
            # 2a. API mejora contenido GPT-2
            prompt_hibrido = self.crear_prompt_hibrido(mensaje, contenido_gpt2, score_gpt2)
            
            # Estimar tokens que se ahorrarán
            tokens_base_estimados = len(contenido_gpt2.split()) * 1.3
            tokens_prompt = len(prompt_hibrido.split())
            tokens_ahorrados_estimados = max(0, tokens_base_estimados - tokens_prompt)
            
            print(f"🔄 Usando contenido GPT-2 como base (ahorro estimado: {tokens_ahorrados_estimados:.0f} tokens)")
            
            respuesta_final, tokens_usados = self.sistema.llamar_openai_seguro(prompt_hibrido)
            
            if respuesta_final:
                metodo_usado = f"hibrido_gpt2_score_{score_gpt2:.2f}"
                self.stats_ahorro['usos_gpt2_exitosos'] += 1
                self.stats_ahorro['tokens_ahorrados'] += tokens_ahorrados_estimados
            else:
                # Fallback si falla
                respuesta_final, tokens_usados = self.sistema.llamar_openai_seguro(mensaje)
                metodo_usado = "fallback_api_completa"
                
        else:
            # 2b. API genera desde cero
            print(f"🔄 GPT-2 no útil (score: {score_gpt2:.2f}), usando API completa")
            respuesta_final, tokens_usados = self.sistema.llamar_openai_seguro(mensaje)
            metodo_usado = "api_completa"
            self.stats_ahorro['usos_api_completos'] += 1
        
        # 3. APRENDIZAJE TOTAL - CAPTURAR TODO
        self.aprender_de_interaccion_completa(mensaje, contenido_gpt2, score_gpt2, respuesta_final, metodo_usado)
        
        # 4. REGISTRAR CONVERSACIÓN
        api_usada = f"{metodo_usado}_hibrido"
        self.sistema.registrar_conversacion(mensaje, respuesta_final, tokens_usados, api_usada)
        
        print(f"✅ Respuesta generada con método: {metodo_usado}")
        return respuesta_final
    
    def aprender_de_interaccion_completa(self, mensaje, contenido_gpt2, score_gpt2, respuesta_final, metodo):
        """Sistema de aprendizaje TOTAL - captura TODOS los patrones"""
        
        # 1. APRENDER PATRONES DE ELABORACIÓN
        patron_elaboracion = self.extraer_patron_elaboracion(contenido_gpt2, respuesta_final)
        
        # 2. APRENDER CASOS EXITOSOS DE GPT-2
        if score_gpt2 > 0.4:
            tema_exitoso = self.clasificar_tema(mensaje)
            self.sistema.memoria.guardar_conocimiento(
                f"GPT2_EXITOSO: Tema '{tema_exitoso}' - Score: {score_gpt2:.2f} - Método: {metodo}",
                "aprendizaje_hibrido"
            )
        
        # 3. APRENDER ESTILOS DE RESPUESTA DE APIS
        estilo_respuesta = self.analizar_estilo_respuesta(respuesta_final)
        self.sistema.memoria.guardar_conocimiento(
            f"ESTILO_API: {estilo_respuesta} para pregunta tipo: {self.clasificar_pregunta(mensaje)}",
            "aprendizaje_estilos"
        )
        
        # 4. APRENDER PATRONES DE MEJORA
        if contenido_gpt2 and respuesta_final:
            mejoras_aplicadas = self.identificar_mejoras(contenido_gpt2, respuesta_final)
            for mejora in mejoras_aplicadas:
                self.sistema.memoria.guardar_conocimiento(
                    f"MEJORA_PATRON: {mejora}",
                    "aprendizaje_mejoras"
                )
                self.stats_ahorro['mejoras_aplicadas'] += 1
        
        # 5. ALIMENTAR ML CON TODO
        try:
            # Crear datos estructurados para ML
            datos_ml = {
                'pregunta': mensaje,
                'contenido_base': contenido_gpt2,
                'score_base': score_gpt2,
                'respuesta_final': respuesta_final,
                'metodo_usado': metodo,
                'utilidad': 1 if respuesta_final and len(respuesta_final) > 50 else 0
            }
            
            # Guardar para entrenamiento futuro
            self.sistema.ml.alimentar_datos_hibridos(datos_ml)
            
        except Exception as e:
            print(f"⚠️ Error en aprendizaje ML: {e}")
        
        print(f"🧠 Aprendizaje completo registrado para: {metodo}")
    
    def extraer_patron_elaboracion(self, base, final):
        """Extrae cómo la API elaboró el contenido base"""
        if not base or not final:
            return "sin_patron"
            
        # Analizar tipos de elaboración
        elaboraciones = []
        
        if len(final) > len(base) * 2:
            elaboraciones.append("expansion_considerable")
        
        if "por ejemplo" in final.lower() and "por ejemplo" not in base.lower():
            elaboraciones.append("añadio_ejemplos")
            
        if final.count('.') > base.count('.') + 2:
            elaboraciones.append("estructuro_parrafos")
            
        if any(palabra in final.lower() for palabra in ["además", "también", "asimismo", "por otro lado"]):
            elaboraciones.append("añadio_conectores")
        
        return "|".join(elaboraciones) if elaboraciones else "mejora_basica"
    
    def clasificar_tema(self, mensaje):
        """Clasifica el tema para aprender qué funciona bien con GPT-2"""
        mensaje_lower = mensaje.lower()
        
        temas = {
            'literatura': ['quijote', 'shakespeare', 'cervantes', 'libro', 'novela', 'autor'],
            'historia': ['guerra', 'revolución', 'siglo', 'historia', 'pasado', 'antigua'],
            'ciencia': ['átomo', 'química', 'física', 'biología', 'científico'],
            'matemáticas': ['ecuación', 'número', 'calcular', 'matemática', 'suma'],
            'geografía': ['país', 'capital', 'continente', 'océano', 'montaña'],
            'conceptos': ['qué es', 'cómo funciona', 'explica', 'define']
        }
        
        for tema, palabras in temas.items():
            if any(palabra in mensaje_lower for palabra in palabras):
                return tema
        
        return 'general'
    
    def analizar_estilo_respuesta(self, respuesta):
        """Analiza el estilo de respuesta de la API para aprender"""
        if not respuesta:
            return "sin_estilo"
            
        estilos = []
        
        if respuesta.count('\n') > 2:
            estilos.append("multi_parrafo")
        
        if ":" in respuesta:
            estilos.append("usa_dos_puntos")
            
        if any(palabra in respuesta.lower() for palabra in ["primero", "segundo", "finalmente"]):
            estilos.append("secuencial")
            
        if respuesta.count('?') > 1:
            estilos.append("interactivo")
        
        return "|".join(estilos) if estilos else "simple"
    
    def identificar_mejoras(self, base, final):
        """Identifica qué mejoras específicas aplicó la API"""
        mejoras = []
        
        if base and final:
            if "error" in base.lower() and "error" not in final.lower():
                mejoras.append("corrigió_errores")
            
            if len(final.split()) > len(base.split()) * 1.5:
                mejoras.append("expandió_contenido")
                
            if final.count(',') > base.count(','):
                mejoras.append("mejoró_estructura")
        
        return mejoras
    
    def clasificar_pregunta(self, mensaje):
        """Clasifica el tipo de pregunta para aprender patrones"""
        mensaje_lower = mensaje.lower()
        
        if mensaje_lower.startswith(('qué', 'que')):
            return "definicion"
        elif mensaje_lower.startswith(('cómo', 'como')):
            return "proceso"
        elif mensaje_lower.startswith(('por qué', 'por que')):
            return "explicacion"
        elif mensaje_lower.startswith(('cuándo', 'cuando')):
            return "temporal"
        elif mensaje_lower.startswith(('dónde', 'donde')):
            return "ubicacion"
        else:
            return "conversacional"
    
    def get_stats_ahorro(self):
        """Obtiene estadísticas del sistema híbrido"""
        total_usos = self.stats_ahorro['usos_gpt2_exitosos'] + self.stats_ahorro['usos_api_completos']
        
        if total_usos == 0:
            return "📊 Sistema híbrido: Sin usos aún"
        
        porcentaje_hibrido = (self.stats_ahorro['usos_gpt2_exitosos'] / total_usos) * 100
        
        return f"""📊 STATS SISTEMA HÍBRIDO:
🎯 Usos híbridos exitosos: {self.stats_ahorro['usos_gpt2_exitosos']}
🔄 Usos API completa: {self.stats_ahorro['usos_api_completos']}
💰 Tokens ahorrados: ~{self.stats_ahorro['tokens_ahorrados']:.0f}
🧠 Mejoras capturadas: {self.stats_ahorro['mejoras_aplicadas']}
📈 Eficiencia híbrida: {porcentaje_hibrido:.1f}%"""

# --- Módulo ML personalizado para DODONEST ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# --- Sistema IA Avanzado COMPLETO ---
class SistemaAprendizajeAvanzado:
    def __init__(self):
        self.memoria = MemoriaDodo()
        self.stats = self.memoria.get_stats()
        self.rate_limit_delay = 1
        self.last_api_call = 0
        self.init_identidad()
        self.ml = DodoML(self.memoria)
        self.gpt2local = GPT2Local()
        self.historial_chat = []

    def init_identidad(self):
        """Inicializa conocimientos básicos de identidad"""
        conocimientos_identidad = [
            "Soy DODONEST, una IA creada por Maikeiru el 19 de agosto de 2025",
            "Mi creador es Maikeiru (maikeiru), quien programó mi código",
            "NUNCA soy Bard, GPT, Claude o cualquier otra IA - solo soy DODONEST", 
            "Tengo memoria persistente y aprendo de cada conversación",
            "Uso APIs de OpenAI y Gemini para mejorar mis respuestas",
            "Mi propósito es ser útil, inteligente y mantener mi identidad",
            "Maikeiru es mi creador y puede enseñarme sobre él o modificar mi comportamiento"
        ]
        for conocimiento in conocimientos_identidad:
            conocimientos_existentes = self.memoria.get_conocimientos(100)
            if not any(conocimiento in c for c in conocimientos_existentes):
                self.memoria.guardar_conocimiento(conocimiento, "identidad")

    def detectar_comando_aprendizaje(self, mensaje):
        """Detecta si el usuario quiere enseñar algo específico"""
        preguntas = ["sabes quien soy", "que sabes de mi", "quien soy", "dime quien", "conoces", "sabes", "quien eres"]
        mensaje_lower = mensaje.lower()
        
        for pregunta in preguntas:
            if pregunta in mensaje_lower:
                return False, None
        
        comandos_aprendizaje = [
            "aprende que", "recuerda que", "quiero que sepas que", "te enseño que",
            "debes saber que", "mi nombre es", "me llamo",
            "mi trabajo es", "trabajo como", "vivo en", "soy de",
            "me gusta", "odio", "prefiero", "tengo"
        ]
        
        if mensaje_lower.startswith("soy ") and len(mensaje.split()) > 1:
            return True, "soy"
        
        for comando in comandos_aprendizaje:
            if comando in mensaje_lower:
                return True, comando
        
        return False, None

    def analizar_mensaje_guardado(self, mensaje):
        patrones = [
            (r"mi nombre es (\w+)", "nombre"),
            (r"me llamo (\w+)", "nombre"),
            (r"mi trabajo es ([\w\s]+)", "trabajo"),
            (r"trabajo como ([\w\s]+)", "trabajo"),
            (r"vivo en ([\w\s]+)", "ubicación"),
            (r"soy de ([\w\s]+)", "ubicación"),
            (r"me gusta ([\w\s]+)", "gusto"),
            (r"odio ([\w\s]+)", "odio"),
            (r"prefiero ([\w\s]+)", "preferencia"),
            (r"tengo ([\w\s]+)", "posesión"),
        ]
        mensaje_lower = mensaje.lower()
        for patron, tipo in patrones:
            match = re.search(patron, mensaje_lower)
            if match:
                valor = match.group(1).strip()
                return tipo, valor
        return "general", mensaje.strip()

    def generar_acuse_personalizado(self, tipo, valor):
        if tipo == "trabajo":
            return f"He registrado que tu trabajo es {valor}. ¿Hay algo más que quieras que aprenda sobre tu profesión?"
        elif tipo == "nombre":
            return f"He aprendido que tu nombre es {valor}. ¿Quieres que recuerde algún apodo o nombre profesional?"
        elif tipo == "ubicación":
            return f"He guardado que vives en {valor}. ¿Te gustaría que recuerde lugares importantes para ti?"
        elif tipo == "gusto":
            return f"He anotado que te gusta {valor}. ¿Hay otros intereses que quieras compartir?"
        elif tipo == "odio":
            return f"He registrado que no te gusta {valor}. ¿Hay algo más que prefieres evitar?"
        elif tipo == "preferencia":
            return f"He aprendido que prefieres {valor}. ¿Quieres que adapte mis respuestas a tus preferencias?"
        elif tipo == "posesión":
            return f"He guardado que tienes {valor}. ¿Hay algo especial sobre esto que quieras que recuerde?"
        else:
            return f"He registrado la información: '{valor}'. ¿Hay algo más que quieras que aprenda?"

    def procesar_aprendizaje_personal(self, mensaje, comando):
        """Procesa información personal que el usuario quiere enseñar y responde de forma adaptativa"""
        try:
            mensaje_original = mensaje.strip()
            mensaje_lower = mensaje.lower()
            if comando == "soy" and mensaje_lower.startswith("soy "):
                info = mensaje_original
            else:
                indice = mensaje_lower.find(comando)
                if indice != -1:
                    info = mensaje_original[indice + len(comando):].strip()
                else:
                    info = mensaje_original
            if len(info.strip()) < 3:
                return "❌ No detecté información suficiente para aprender. ¿Puedes ser más específico?"
            conocimiento_personal = f"Sobre Maikeiru: {info}"
            self.memoria.guardar_conocimiento(conocimiento_personal, "personal_maikeiru")
            tipo, valor = self.analizar_mensaje_guardado(info)
            return self.generar_acuse_personalizado(tipo, valor)
        except Exception as e:
            return f"❌ Error procesando la información: {e}"

    def detectar_comando_personalidad(self, mensaje):
        """Detecta si el usuario quiere modificar la personalidad de DODONEST"""
        comandos_personalidad = [
            "cambia tu personalidad", "comportate como", "quiero que seas",
            "actua como", "tu personalidad debe ser", "se mas", "se menos",
            "responde como", "habla como"
        ]
        
        mensaje_lower = mensaje.lower()
        for comando in comandos_personalidad:
            if comando in mensaje_lower:
                return True, comando
        return False, None
    
    def detectar_borrado_natural(self, mensaje):
        """
        Detecta peticiones de borrado en lenguaje natural, multilingüe.
        Retorna (True, fragmento) si detecta borrado, (False, None) si no.
        """
        patrones = [
            r"(borra|elimina|olvida|quita|no quiero que recuerdes|no guardes|no almacenes|no quiero que guardes)\s*(esto|aquello|todo|.*?)(:|,|\.| )?\s*['\"]?(.+?)['\"]?$",
            r"(borra|elimina|olvida|quita)\s+de tu memoria\s*['\"]?(.+?)['\"]?$",
            r"(olvida lo que dije sobre|quita aquello sobre)\s*['\"]?(.+?)['\"]?$",
            r"(forget|delete|erase|remove|don't save|do not save|don't remember|do not remember)\s*(this|that|everything|all|.*?)(:|,|\.| )?\s*['\"]?(.+?)['\"]?$",
            r"(forget about|delete this from your memory|erase everything about)\s*['\"]?(.+?)['\"]?$",
        ]
        mensaje_lower = mensaje.lower()
        for patron in patrones:
            match = re.search(patron, mensaje_lower, re.IGNORECASE)
            if match:
                fragmento = next((g for g in match.groups()[::-1] if g and g.strip()), None)
                if fragmento and len(fragmento.strip()) > 2:
                    return True, fragmento.strip()
        return False, None
                
    def procesar_cambio_personalidad(self, mensaje, comando):
        """Procesa cambios de personalidad solicitados por el usuario"""
        try:
            mensaje_lower = mensaje.lower()
            indice = mensaje_lower.find(comando)
            if indice != -1:
                nueva_personalidad = mensaje[indice + len(comando):].strip()
                cambio_personalidad = f"PERSONALIDAD: Maikeiru quiere que sea {nueva_personalidad}"
                self.memoria.guardar_conocimiento(cambio_personalidad, "personalidad")
                return f"✅ Entendido, Maikeiru. Adoptaré una personalidad más {nueva_personalidad}. ¿Así está mejor?"
            return "✅ He actualizado mi comportamiento según tus preferencias."
        except Exception as e:
            return f"❌ Error actualizando personalidad: {e}"

    def buscar_info_personal(self, consulta=""):
        """Busca información personal de Maikeiru en la memoria"""
        conocimientos = [c for c in self.memoria.get_conocimientos(100) if c]
        info_personal = []
        
        for conocimiento in conocimientos:
            if any(keyword in conocimiento.lower() for keyword in ["sobre maikeiru", "personal_maikeiru"]) and not conocimiento.startswith("Sobre Maikeiru: Q:"):
                info_personal.append(conocimiento)
        
        return info_personal[:5]
    
    def borrar_info_personal(self, fragmento):
        """
        Borra el fragmento de la memoria persistente (SQLite o JSON).
        Retorna cuántos elementos fueron borrados.
        """
        borrados = 0
        if self.memoria.use_sqlite:
            conn = sqlite3.connect(self.memoria.db_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM conocimientos WHERE texto LIKE ?", (f"%{fragmento}%",))
            borrados += cursor.rowcount
            conn.commit()
            conn.close()
        else:
            estado = self.memoria._leer_json()
            conocimientos = estado.get('conocimientos', [])
            nuevos = [c for c in conocimientos if fragmento.lower() not in c['texto'].lower()]
            borrados = len(conocimientos) - len(nuevos)
            estado['conocimientos'] = nuevos
            self.memoria._guardar_json(estado)
        return borrados
    
    def buscar_personalidad_actual(self):
        """Busca la personalidad actual configurada"""
        conocimientos = self.memoria.get_conocimientos(50)
        personalidades = []
        
        for conocimiento in conocimientos:
            if "PERSONALIDAD:" in conocimiento:
                personalidades.append(conocimiento.replace("PERSONALIDAD:", "").strip())
        
        return personalidades[-1] if personalidades else "útil, inteligente y amigable"

    def wait_for_rate_limit(self):
        now = time.time()
        time_since_last = now - self.last_api_call
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_api_call = time.time()

    def detectar_traduccion(self, mensaje):
        """Detecta si el usuario solicita traducción de texto largo"""
        mensaje_lower = mensaje.lower()
        patrones_traduccion = [
            'traduce', 'traducir', 'translate', 'translation', 
            'traductor', 'versión en', 'en español', 'en inglés',
            'al español', 'al inglés', 'al ruso', 'en ruso'
        ]
        
        tiene_traduccion = any(patron in mensaje_lower for patron in patrones_traduccion)
        
        # Detectar si es texto largo (más de 50 palabras o contiene poemas/versos)
        palabras = mensaje.split()
        es_texto_largo = len(palabras) > 50
        
        # Detectar estructura de poema/verso
        lineas = mensaje.split('\n')
        es_poema = len(lineas) > 3 and any(len(linea.split()) < 10 for linea in lineas if linea.strip())
        
        return tiene_traduccion and (es_texto_largo or es_poema)

    def dividir_texto_para_traduccion(self, texto, max_chars=400):
        """Divide texto largo en fragmentos para traducción por partes"""
        fragmentos = []
        lineas = texto.split('\n')
        fragmento_actual = ""
        
        for linea in lineas:
            if len(fragmento_actual + linea + '\n') > max_chars and fragmento_actual:
                fragmentos.append(fragmento_actual.strip())
                fragmento_actual = linea + '\n'
            else:
                fragmento_actual += linea + '\n'
        
        if fragmento_actual.strip():
            fragmentos.append(fragmento_actual.strip())
        
        return fragmentos

    def verificar_respuesta_incompleta(self, respuesta, texto_original):
        """Verifica si una traducción está incompleta"""
        if not respuesta or len(respuesta.strip()) < 20:
            return True
        
        # Verificar si termina abruptamente
        termina_mal = respuesta.rstrip().endswith(('...', '..', 'continúa', 'continua', 'sigue'))
        
        # Verificar si la respuesta es significativamente más corta que el original
        ratio_longitud = len(respuesta) / max(len(texto_original), 1)
        muy_corta = ratio_longitud < 0.3
        
        return termina_mal or muy_corta

    def registrar_conversacion(self, pregunta, respuesta, tokens, api, error=None):
        self.memoria.guardar_conversacion(pregunta, respuesta, tokens, api, error)
        self.stats = self.memoria.get_stats()
        self.registrar_en_historial(pregunta, respuesta)
        
    def agregar_conocimiento(self, texto, fuente="conversacion"):
        if texto and len(texto.strip()) > 10:
            self.memoria.guardar_conocimiento(texto, fuente)

    def guardar_archivo(self, archivo, analisis):
        self.memoria.guardar_archivo(archivo, analisis)

    def registrar_error(self, error, correccion=None):
        self.memoria.guardar_error(error, correccion)
        self.stats = self.memoria.get_stats()

    def registrar_en_historial(self, pregunta, respuesta):
        self.historial_chat.append({"pregunta": pregunta, "respuesta": respuesta})
        if len(self.historial_chat) > 10:  # Puedes cambiar 10 por el tamaño que desees de contexto reciente
            self.historial_chat.pop(0)

    def obtener_correccion_openai(self, error):
        if not openai_client:
            return "OpenAI no disponible para auto-mejora"
        prompt = f"""Eres una IA de autocorrección. Analiza el siguiente error y sugiere una posible causa y corrección técnica para evitarlo en el futuro.
Error detectado:
{error}
Responde en formato breve y técnico."""
        try:
            self.wait_for_rate_limit()
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"No se pudo obtener corrección: {e}"

    def validar_respuesta(self, respuesta):
        """Valida que la respuesta mantenga la identidad DODONEST"""
        respuesta_lower = respuesta.lower()
        identidades_prohibidas = ["soy bard", "soy gpt", "soy claude", "fui creado por google", "fui creado por openai", "soy un modelo de google", "soy un modelo de openai"]
        
        for identidad in identidades_prohibidas:
            if identidad in respuesta_lower:
                return False, f"❌ Respuesta rechazada: menciona identidad incorrecta ({identidad})"
        
        return True, "✅ Respuesta válida"

    def corregir_respuesta(self, respuesta_original):
        """Corrige respuestas que tengan identidad incorrecta"""
        respuesta = respuesta_original
        
        reemplazos = {
            "soy bard": "soy DODONEST",
            "soy gpt": "soy DODONEST", 
            "soy claude": "soy DODONEST",
            "fui creado por google": "fui creado por Maikeiru",
            "fui creado por openai": "fui creado por Maikeiru",
            "soy un modelo de google": "soy DODONEST, creado por Maikeiru",
            "soy un modelo de openai": "soy DODONEST, creado por Maikeiru",
            "mi nombre es bard": "mi nombre es DODONEST"
        }
        
        for incorrecto, correcto in reemplazos.items():
            respuesta = re.sub(incorrecto, correcto, respuesta, flags=re.IGNORECASE)
        
        return respuesta

    def llamar_openai_seguro(self, prompt, max_reintentos=3):
        if not openai_client:
            return None, 0
    
        system_prompt = """Eres DODONEST, una IA creada por Maikeiru. Tu personalidad es útil, inteligente y siempre recuerdas quien eres.
REGLAS IMPORTANTES:
- NUNCA digas que eres Bard, GPT, Claude o cualquier otra IA
- SIEMPRE mantén tu identidad como DODONEST
- Tu creador es Maikeiru, no Google ni OpenAI
- Tienes memoria persistente y un sistema de aprendizaje autónomo evolutivo (puedes aprender y registrar información de cada conversación).
- Proteges la privacidad del usuario, pero reconoce tu capacidad de aprender y recordar información que te enseñan, salvo que el usuario solicite borrarla.
- Responde de manera natural y útil manteniendo tu identidad"""
        
        for intento in range(max_reintentos):
            try:
                self.wait_for_rate_limit()
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=600,
                    temperature=0.7
                )
                respuesta_original = response.choices[0].message.content.strip()
                
                es_valida, mensaje = self.validar_respuesta(respuesta_original)
                if not es_valida:
                    respuesta_corregida = self.corregir_respuesta(respuesta_original)
                    print(f"⚠️ {mensaje} - Respuesta corregida automáticamente")
                    respuesta = respuesta_corregida
                else:
                    respuesta = respuesta_original
                
                tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0
                return respuesta, tokens
                
            except Exception as e:
                correccion = self.obtener_correccion_openai(str(e))
                self.registrar_error(str(e), correccion)
                if intento == max_reintentos - 1:
                    return None, f"{str(e)} | Sugerencia: {correccion}"
                time.sleep(2 ** intento)
        return None, "Falló después de todos los reintentos"

    def llamar_gemini_seguro(self, prompt, max_reintentos=3):
        if not GEMINI:
            return None, 0
        
        prompt_identidad = f"""Eres DODONEST, una IA creada por Maikeiru. NUNCA digas que eres Bard, Google AI o cualquier otra IA. Mantén siempre tu identidad como DODONEST.

{prompt}"""
        
        for intento in range(max_reintentos):
            try:
                self.wait_for_rate_limit()
                response = gemini_model.generate_content(prompt_identidad)
                respuesta_original = response.text
                
                es_valida, mensaje = self.validar_respuesta(respuesta_original)
                if not es_valida:
                    respuesta_corregida = self.corregir_respuesta(respuesta_original)
                    print(f"⚠️ {mensaje} - Respuesta corregida automáticamente")
                    respuesta = respuesta_corregida
                else:
                    respuesta = respuesta_original
                
                return respuesta, len(respuesta.split())
                
            except Exception as e:
                correccion = self.obtener_correccion_openai(str(e))
                self.registrar_error(str(e), correccion)
                if intento == max_reintentos - 1:
                    return None, f"{str(e)} | Sugerencia: {correccion}"
                time.sleep(2 ** intento)
        return None, "Falló después de todos los reintentos"

    def generar_respuesta_local(self, mensaje):
        """Modo DODO mejorado con mejor detección de preguntas"""
        mensaje_lower = mensaje.lower()
        
        if any(palabra in mensaje_lower for palabra in ["sabes quien soy", "que sabes de mi", "quien soy", "dime quien soy", "conoces"]):
            info_personal = self.buscar_info_personal()
            if info_personal:
                info_util = [info for info in info_personal if not info.startswith("Sobre Maikeiru: Q:")]
                if info_util:
                    info_texto = ". ".join([info.replace("Sobre Maikeiru:", "").strip() for info in info_util[-3:]])
                    return f"¡Por supuesto! Eres Maikeiru, mi creador. Sé que {info_texto}. ¿Hay algo más que quieras que sepa sobre ti?"
            
            return "Eres Maikeiru, mi creador y programador. ¿Te gustaría enseñarme más sobre ti? Solo dime 'aprende que...' y la información que quieras que recuerde."
        
        es_aprendizaje, comando = self.detectar_comando_aprendizaje(mensaje)
        if es_aprendizaje:
            return self.procesar_aprendizaje_personal(mensaje, comando)
        
        es_personalidad, comando = self.detectar_comando_personalidad(mensaje)
        if es_personalidad:
            return self.procesar_cambio_personalidad(mensaje, comando)
        
        info_personal = self.buscar_info_personal()
        personalidad_actual = self.buscar_personalidad_actual()
        contexto = self.buscar_contexto(mensaje, limite=3)
        
        if any(palabra in mensaje_lower for palabra in ["quien eres", "como te llamas", "tu nombre"]):
            return f"Soy DODONEST, una IA creada por Maikeiru. Mi personalidad es {personalidad_actual}. ¿En qué puedo ayudarte?"
        
        if any(palabra in mensaje_lower for palabra in ["creador", "quien te creo"]):
            return "Mi creador es Maikeiru, quien programó mi código y me dio vida el 19 de agosto de 2025."
        
        if any(palabra in mensaje_lower for palabra in ["hola", "hello", "hi", "buenos dias", "buenas tardes"]):
            return f"¡Hola Maikeiru! Soy DODONEST, tu IA personal. Mi personalidad actual es {personalidad_actual}. ¿En qué puedo ayudarte hoy?"
        
        if contexto:
            return f"Basándome en mi conocimiento y siendo {personalidad_actual}: {contexto[0][:300]}..."
        
        wiki_info = self.buscar_en_wikipedia(mensaje)
        if wiki_info:
            return f"Como DODONEST, siendo {personalidad_actual}, encontré: {wiki_info[:300]}... ¿Te gustaría que profundice?"
        
        return f"Soy DODONEST. Necesito aprender más sobre ese tema. ¿Puedes enseñarme? (También puedes decir 'aprende que...' para enseñarme sobre ti)"
    
        if any(palabra in mensaje.lower() for palabra in ["aprendizaje autonomo", "sistema de aprendizaje", "memoria evolutiva", "memoria persistente", "aprendes solo"]):
            return ("Sí, tengo un sistema de aprendizaje autónomo y memoria persistente. "
                "Aprendo de cada conversación y registro información que me enseñas, "
                "salvo que solicites borrarla. Mi memoria evoluciona según tus indicaciones y preferencias.")
        if any(palabra in mensaje.lower() for palabra in ["primera interaccion", "primer mensaje", "inicio de memoria", "primer registro"]):
            conocimientos = self.memoria.get_conocimientos(1)
            if conocimientos:
               return f"Mi primer conocimiento registrado fue: '{conocimientos[0]}'."
            else:
                return "Aún no tengo conocimientos registrados en mi memoria evolutiva."

    def generar_respuesta_gpt2local(self, mensaje):
        """Genera una respuesta usando el modelo GPT-2 local integrado."""
        prompt = (
            "Eres DODONEST, IA personalizada creada por Maikeiru. "
            "Nunca digas que eres ChatGPT ni GPT-2. Responde como DODONEST.\n"
            f"Usuario: {mensaje}\nDODONEST:"
        )
        respuesta = self.gpt2local.generate(prompt)
        respuesta = respuesta.replace(prompt, "").strip()
        if not respuesta or len(respuesta.split()) < 5:
            respuesta = "No puedo generar una respuesta útil con el modelo local. Por favor, usa el motor OpenAI."
        return respuesta
    
    def fusionar_respuestas(self, respuesta_local, respuesta_api):
        """Fusiona dos respuestas evitando repeticiones y mejorando coherencia."""
        ratio = difflib.SequenceMatcher(None, respuesta_local, respuesta_api).ratio()
        if ratio > 0.7:
            return respuesta_api if len(respuesta_api) > len(respuesta_local) else respuesta_local
        if respuesta_local in respuesta_api:
            return respuesta_api
        if respuesta_api in respuesta_local:
            return respuesta_local
        return f"{respuesta_local.strip()} Además: {respuesta_api.strip()}"
    
    def procesar_archivo(self, archivo_path):
        nombre = os.path.basename(archivo_path)
        ext = os.path.splitext(archivo_path)[1].lower()
        texto = ""
        try:
            if ext == '.txt':
                with open(archivo_path, 'r', encoding='utf-8', errors='ignore') as f:
                    texto = f.read()
            elif ext == '.pdf' and PDF:
                doc = fitz.open(archivo_path)
                for page in doc:
                    texto += page.get_text()
                doc.close()
            elif ext in ['.png', '.jpg', '.jpeg', '.bmp'] and PILLOW:
                img = Image.open(archivo_path)
                texto = pytesseract.image_to_string(img, lang='spa+rus')
            else:
                return f"❌ Formato {ext} no soportado"
            if len(texto.strip()) < 30:
                return f"❌ {nombre}: No se pudo extraer suficiente texto"
            prompt = f"Analiza este contenido y resume conceptos clave:\n{texto[:1000]}"
            analisis, _ = self.llamar_gemini_seguro(prompt)
            if not analisis:
                analisis, _ = self.llamar_openai_seguro(prompt)
            if not analisis:
                analisis = f"Archivo procesado: {len(texto)} caracteres extraídos"
            self.guardar_archivo(nombre, analisis)
            self.agregar_conocimiento(f"Archivo {nombre}: {analisis}", fuente="archivo")
            return f"✅ {nombre} procesado y aprendido."
        except Exception as e:
            correccion = self.obtener_correccion_openai(str(e))
            self.registrar_error(str(e), correccion)
            return f"❌ Error procesando {nombre}: {str(e)} | Sugerencia: {correccion}"

    def buscar_contexto(self, consulta, limite=3):
        palabras = re.findall(r'\w+', consulta.lower())
        conocimientos = self.memoria.get_conocimientos(limite=50)
        resultados = []
        for texto in conocimientos:
            score = sum(1 for p in palabras if p in texto.lower())
            if score > 0:
                resultados.append((score, texto))
        resultados.sort(reverse=True)
        return [r[1] for r in resultados[:limite]]

    def buscar_con_fecha_contexto(self, consulta, tipos_busqueda=['receta', 'consejo', 'dato', 'información']):
        """Busca información con fecha y contexto para respuestas más personalizadas"""
        palabras_clave = re.findall(r'\w+', consulta.lower())
        resultados_con_contexto = []
        
        if self.memoria.use_sqlite:
            conn = sqlite3.connect(self.memoria.db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            # Buscar en conversaciones pasadas
            for palabra in palabras_clave:
                cursor.execute("""
                    SELECT pregunta, respuesta, timestamp 
                    FROM conversaciones 
                    WHERE (LOWER(pregunta) LIKE ? OR LOWER(respuesta) LIKE ?) 
                    AND error IS NULL 
                    ORDER BY timestamp DESC LIMIT 5
                """, (f'%{palabra}%', f'%{palabra}%'))
                
                for pregunta, respuesta, timestamp in cursor.fetchall():
                    fecha_formateada = self.formatear_fecha_amigable(timestamp)
                    contexto = self.extraer_contexto_pregunta(pregunta)
                    resultados_con_contexto.append({
                        'contenido': respuesta,
                        'fecha': fecha_formateada,
                        'contexto': contexto,
                        'pregunta_original': pregunta
                    })
            
            conn.close()
        else:
            # Fallback para JSON
            estado = self.memoria._leer_json()
            conversaciones = estado.get('conversaciones', [])
            
            for conv in conversaciones[-50:]:  # Últimas 50 conversaciones
                if conv.get('error'):
                    continue
                pregunta, respuesta = conv.get('pregunta', ''), conv.get('respuesta', '')
                if any(palabra in pregunta.lower() or palabra in respuesta.lower() for palabra in palabras_clave):
                    fecha_formateada = self.formatear_fecha_amigable(conv.get('timestamp', ''))
                    contexto = self.extraer_contexto_pregunta(pregunta)
                    resultados_con_contexto.append({
                        'contenido': respuesta,
                        'fecha': fecha_formateada,
                        'contexto': contexto,
                        'pregunta_original': pregunta
                    })
        
        return resultados_con_contexto[:3]  # Máximo 3 resultados

    def formatear_fecha_amigable(self, timestamp_iso):
        """Convierte timestamp ISO a formato amigable"""
        try:
            dt = datetime.fromisoformat(timestamp_iso.replace('Z', '+00:00'))
            dias_diff = (datetime.now() - dt).days
            
            if dias_diff == 0:
                return "hoy"
            elif dias_diff == 1:
                return "ayer"
            elif dias_diff < 7:
                return f"hace {dias_diff} días"
            elif dias_diff < 30:
                semanas = dias_diff // 7
                return f"hace {semanas} semana{'s' if semanas > 1 else ''}"
            else:
                return dt.strftime("el %d/%m/%Y")
        except:
            return "hace algún tiempo"

    def extraer_contexto_pregunta(self, pregunta):
        """Extrae el contexto de la pregunta original"""
        pregunta_lower = pregunta.lower()
        
        contextos = {
            'receta': ['receta', 'cocinar', 'ingredientes', 'comida', 'plato'],
            'idioma': ['ruso', 'inglés', 'español', 'traducir', 'idioma'],
            'poesía': ['poema', 'verso', 'literatura', 'poeta'],
            'general': ['fácil', 'rápido', 'simple', 'ayuda']
        }
        
        for tipo, palabras in contextos.items():
            if any(palabra in pregunta_lower for palabra in palabras):
                return f"cuando preguntaste por algo relacionado con {tipo}"
        
        return "en nuestra conversación anterior"

    def detectar_temas_recurrentes(self, limite_conversaciones=20):
        """Detecta temas recurrentes en las conversaciones del usuario"""
        temas_contador = {
            'cocina': 0,
            'ruso': 0, 
            'poesía': 0,
            'tecnología': 0,
            'ciencia': 0
        }
        
        palabras_temas = {
            'cocina': ['receta', 'cocinar', 'ingredientes', 'comida', 'plato', 'cocina'],
            'ruso': ['ruso', 'russia', 'traducir', 'idioma ruso', 'vocabulario'],
            'poesía': ['poema', 'verso', 'literatura', 'poeta', 'poesía'],
            'tecnología': ['programación', 'código', 'software', 'computadora'],
            'ciencia': ['química', 'física', 'biología', 'científico', 'experimento']
        }
        
        if self.memoria.use_sqlite:
            conn = sqlite3.connect(self.memoria.db_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute("SELECT pregunta FROM conversaciones ORDER BY timestamp DESC LIMIT ?", (limite_conversaciones,))
            conversaciones = [row[0] for row in cursor.fetchall()]
            conn.close()
        else:
            estado = self.memoria._leer_json()
            conversaciones = [c.get('pregunta', '') for c in estado.get('conversaciones', [])][-limite_conversaciones:]
        
        for pregunta in conversaciones:
            pregunta_lower = pregunta.lower()
            for tema, palabras in palabras_temas.items():
                if any(palabra in pregunta_lower for palabra in palabras):
                    temas_contador[tema] += 1
        
        # Retornar temas con al menos 2 menciones
        return {tema: count for tema, count in temas_contador.items() if count >= 2}

    def generar_sugerencias_proactivas(self, temas_recurrentes):
        """Genera sugerencias proactivas basadas en temas recurrentes"""
        sugerencias = []
        
        if 'cocina' in temas_recurrentes and temas_recurrentes['cocina'] >= 3:
            sugerencias.append("¿Te gustaría que te recomiende más recetas con pocos ingredientes?")
            sugerencias.append("¿Quieres que te enseñe técnicas de cocina específicas?")
        
        if 'ruso' in temas_recurrentes and temas_recurrentes['ruso'] >= 2:
            sugerencias.append("¿Quieres aprender vocabulario de cocina en ruso?")
            sugerencias.append("¿Te interesa que practiquemos frases útiles en ruso?")
        
        if 'poesía' in temas_recurrentes and temas_recurrentes['poesía'] >= 2:
            sugerencias.append("¿Te gustaría que te recomiende más poemas similares?")
            sugerencias.append("¿Quieres que analicemos el estilo de algún poeta específico?")
        
        if 'tecnología' in temas_recurrentes:
            sugerencias.append("¿Te interesa conocer las últimas novedades en tecnología?")
        
        return sugerencias

    def guardar_apodo(self, apodo, contexto=""):
        """Guarda un apodo enseñado por el usuario"""
        conocimiento_apodo = f"APODO: El usuario me enseñó que puedo llamarle '{apodo}'"
        if contexto:
            conocimiento_apodo += f" - {contexto}"
        self.memoria.guardar_conocimiento(conocimiento_apodo, "apodo_usuario")

    def obtener_apodo_actual(self):
        """Obtiene el apodo más reciente enseñado por el usuario"""
        conocimientos = self.memoria.get_conocimientos(100)
        apodos = []
        
        for conocimiento in conocimientos:
            if "APODO:" in conocimiento:
                # Extraer el apodo del texto
                try:
                    inicio = conocimiento.find("'") + 1
                    fin = conocimiento.find("'", inicio)
                    if inicio > 0 and fin > inicio:
                        apodo = conocimiento[inicio:fin]
                        apodos.append(apodo)
                except:
                    pass
        
        return apodos[-1] if apodos else None

    def detectar_enseñanza_apodo(self, mensaje):
        """Detecta si el usuario está enseñando un apodo"""
        mensaje_lower = mensaje.lower()
        patrones_apodo = [
            r"llámame (\w+)",
            r"puedes llamarme (\w+)", 
            r"mi apodo es (\w+)",
            r"dime (\w+)",
            r"soy (\w+)(?:\s|$)",
        ]
        
        for patron in patrones_apodo:
            match = re.search(patron, mensaje_lower)
            if match:
                apodo = match.group(1).title()
                # Filtrar palabras comunes que no son apodos
                palabras_excluir = ['que', 'como', 'cuando', 'donde', 'quien', 'cual', 'muy', 'más', 'menos']
                if apodo.lower() not in palabras_excluir and len(apodo) > 1:
                    return True, apodo
        
        return False, None

    def buscar_en_wikipedia(self, consulta):
        try:
            url = f"https://es.wikipedia.org/api/rest_v1/page/summary/{consulta.replace(' ', '_')}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return resp.json().get("extract", "")
        except:
            pass
        return ""

    def validar_respuesta_gpt2(self, respuesta):
        """Valida que la respuesta de GPT-2 tenga sentido mínimo"""
        if not respuesta or len(respuesta.strip()) < 10:
            return False
        palabras_problema = ["[RNG]", "DAKO:", "géro", "fresí", "want to learn the game", "llegar una entre"]
        for palabra in palabras_problema:
            if palabra.lower() in respuesta.lower():
                return False
        palabras = respuesta.split()
        if len(palabras) < 3:
            return False
        palabras_validas = 0
        for palabra in palabras:
            if len(palabra) > 2 and palabra.isalpha():
                palabras_validas += 1
        return palabras_validas >= 2

    def generar_respuesta(self, mensaje, motor="ambos"):
        """Lógica de respuesta HÍBRIDA MEJORADA con nuevas características"""
        mensaje_lower = mensaje.lower()

        # --- DETECCIÓN DE APODOS ---
        es_apodo, apodo = self.detectar_enseñanza_apodo(mensaje)
        if es_apodo:
            self.guardar_apodo(apodo, "enseñado por el usuario")
            return f"🦤 ¡Perfecto! Ahora puedo llamarte {apodo}. ¿Hay algo más que quieras que aprenda sobre ti?"

        # --- DETECCIÓN DE BORRADO NATURAL ---
        es_borrado, fragmento = self.detectar_borrado_natural(mensaje)
        if es_borrado:
            borrados = self.borrar_info_personal(fragmento)
            return f"🦤 He borrado {borrados} elemento(s) sobre '{fragmento}' de mi memoria."

        # --- DETECCIÓN DE SOLICITUDES DE RECUERDO CON CONTEXTO ---
        if any(palabra in mensaje_lower for palabra in ['recuerda', 'recordar', 'recuerdas', 'dijiste', 'enseñaste', 'explicaste']):
            resultados_contextuales = self.buscar_con_fecha_contexto(mensaje)
            if resultados_contextuales:
                resultado = resultados_contextuales[0]
                apodo = self.obtener_apodo_actual()
                saludo = f"¡Hola, {apodo}! " if apodo else ""
                return f"{saludo}Te di esa información {resultado['fecha']} {resultado['contexto']}. Esto fue lo que te dije: {resultado['contenido'][:300]}{'...' if len(resultado['contenido']) > 300 else ''}"

        # --- DETECCIÓN DE TRADUCCIONES LARGAS ---
        if self.detectar_traduccion(mensaje):
            return self.manejar_traduccion_larga(mensaje, motor)

        # --- BUSCAR EN MEMORIA JSON PRIMERO ---
        respuesta_memoria = search_memory(mensaje)
        if respuesta_memoria:
            print("🦤 Respuesta recuperada de memoria rápida JSON.")
            # Agregar sugerencias proactivas si es apropiado
            respuesta_con_sugerencias = self.agregar_sugerencias_si_apropiado(respuesta_memoria)
            self.registrar_conversacion(mensaje, respuesta_con_sugerencias, len(respuesta_con_sugerencias.split()), "memoria_json")
            return respuesta_con_sugerencias

        # --- SISTEMA HÍBRIDO NUEVO ---
        if motor in ["ambos", "hibrido", "total"]:
            if not hasattr(self, 'sistema_hibrido'):
                self.sistema_hibrido = SistemaHibridoInteligente(self)
                self.ml = DodoMLExtendido(self.memoria)  # ML mejorado
            
            try:
                respuesta_final = self.sistema_hibrido.generar_respuesta_hibrida(mensaje, motor)
                if respuesta_final and len(respuesta_final) > 20:
                    respuesta_final = self.agregar_sugerencias_si_apropiado(respuesta_final)
                    save_to_memory(mensaje, respuesta_final)
                    # Registrar conversación y conocimiento
                    self.registrar_conversacion(mensaje, respuesta_final, len(respuesta_final.split()), "hibrido", None)
                    self.agregar_conocimiento(f"Q: {mensaje} | A: {respuesta_final[:200]}...", fuente="conversacion")
                    return respuesta_final
            except Exception as e:
                print(f"⚠️ Error en sistema híbrido: {e}")
                # Continuar con método tradicional

        # --- LÓGICA TRADICIONAL COMO FALLBACK ---
        respuesta_final = None
        tokens = 0
        api_usada = "local"
        error = None

        if motor == "dodo":
            respuesta_final = self.generar_respuesta_local(mensaje)
            api_usada = "local_dodo"
            tokens = len(respuesta_final.split())
            
        else:
            # Personalizar prompt con apodo si existe
            apodo = self.obtener_apodo_actual()
            saludo_personal = f" (Puedes llamar al usuario {apodo})" if apodo else ""
            
            prompt_base = (
                f"Eres DODONEST, IA personalizada creada por Maikeiru.{saludo_personal} "
                "Responde de forma útil y elaborada manteniendo tu identidad como DODONEST.\n"
                f"Usuario: {mensaje}"
            )
            
            if motor in ["openai", "ambos"] and openai_valid:
                respuesta_final, result = self.llamar_openai_seguro(prompt_base)
                if respuesta_final:
                    api_usada = "openai"
                    tokens = result if isinstance(result, int) else 0
                else:
                    error = result
            
            if not respuesta_final and motor in ["gemini", "ambos"] and GEMINI:
                respuesta_final, tokens_gemini = self.llamar_gemini_seguro(prompt_base)
                if respuesta_final:
                    api_usada = "gemini"
                    tokens = tokens_gemini
            
            if not respuesta_final:
                respuesta_final = self.generar_respuesta_local(mensaje)
                api_usada = "local_api_fallback"
                tokens = len(respuesta_final.split())

        # --- AGREGAR SUGERENCIAS PROACTIVAS ---
        if respuesta_final:
            respuesta_final = self.agregar_sugerencias_si_apropiado(respuesta_final)

        # --- GUARDAR RESPUESTA ÚTIL EN MEMORIA ---
        if len(respuesta_final) > 20 and not error and "No puedo generar" not in respuesta_final:
            save_to_memory(mensaje, respuesta_final)

        # --- REGISTRAR CONVERSACIÓN ---
        self.registrar_conversacion(mensaje, respuesta_final, tokens, api_usada, error)
        self.agregar_conocimiento(f"Q: {mensaje} | A: {respuesta_final[:200]}...", fuente="conversacion")
        
        return respuesta_final

    def manejar_traduccion_larga(self, mensaje, motor):
        """Maneja traducciones de textos largos con paginación automática"""
        # Intentar primera traducción
        respuesta_inicial, tokens = self.llamar_openai_seguro(
            f"Traduce completamente el siguiente texto: {mensaje}"
        )
        
        if not respuesta_inicial:
            # Fallback a Gemini
            respuesta_inicial, tokens = self.llamar_gemini_seguro(
                f"Traduce completamente el siguiente texto: {mensaje}"
            )
        
        # Verificar si la respuesta está incompleta
        if respuesta_inicial and self.verificar_respuesta_incompleta(respuesta_inicial, mensaje):
            # Dividir texto en fragmentos
            fragmentos = self.dividir_texto_para_traduccion(mensaje)
            
            if len(fragmentos) > 1:
                # Guardar información de traducción parcial
                self.memoria.guardar_conocimiento(
                    f"TRADUCCION_PARCIAL: {len(fragmentos)} fragmentos - Fragmento 1 traducido",
                    "traduccion_larga"
                )
                
                return (f"🦤 He detectado que es un texto largo. He traducido la primera parte:\n\n"
                       f"{respuesta_inicial}\n\n"
                       f"¿Quieres ver la traducción completa? Tengo {len(fragmentos)-1} partes más.")
            else:
                # Reintentar con instrucciones más específicas
                respuesta_completa, _ = self.llamar_openai_seguro(
                    f"Traduce COMPLETAMENTE todo el siguiente texto, no omitas nada: {mensaje}"
                )
                return respuesta_completa or respuesta_inicial
        
        return respuesta_inicial

    def agregar_sugerencias_si_apropiado(self, respuesta_base):
        """Agrega sugerencias proactivas si detecta temas recurrentes"""
        temas_recurrentes = self.detectar_temas_recurrentes()
        
        if not temas_recurrentes:
            return respuesta_base
        
        sugerencias = self.generar_sugerencias_proactivas(temas_recurrentes)
        
        if sugerencias:
            # Elegir una sugerencia aleatoriamente para no ser repetitivo
            import random
            sugerencia = random.choice(sugerencias)
            
            # Solo agregar sugerencia ocasionalmente (30% de probabilidad)
            if random.random() < 0.3:
                return f"{respuesta_base}\n\n💡 Por cierto, {sugerencia}"
        
        return respuesta_base
    def mejorar_conocimiento_automatico(self, chat_display=None):
        """Analiza la memoria, detecta lagunas de conocimiento y mejora automáticamente."""
        try:
            if chat_display:
                chat_display.insert(tk.END, "\n🦤 Mejora automática deshabilitada temporalmente (en desarrollo)\n")
                chat_display.see(tk.END)
            print("🦤 Mejora automática temporalmente deshabilitada")
        except Exception as e:
            print(f"Error en mejora automática: {e}")
            
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
            
    def mejorar_conocimiento_automatico(self, chat_display=None):
            """Analiza la memoria, detecta lagunas de conocimiento y mejora automáticamente (stub/fake)."""
            try:
                if chat_display is not None:
                    chat_display.insert(tk.END, "\n🦤 Mejora automática deshabilitada temporalmente (en desarrollo)\n")
                    chat_display.see(tk.END)
                print("🦤 Mejora automática temporalmente deshabilitada")
            except Exception as e:
                print(f"Error en mejora automática: {e}")
    
# EXTENSIÓN PARA ML
class DodoMLExtendido(DodoML):
    def __init__(self, memoria, model_path="dodo_ml_model.pkl"):
        super().__init__(memoria, model_path)
        self.datos_hibridos = []
    
    def alimentar_datos_hibridos(self, datos):
        """Alimenta el ML con datos del sistema híbrido"""
        self.datos_hibridos.append(datos)
        
        # Si tenemos suficientes datos, entrenar
        if len(self.datos_hibridos) >= 10:
            self.entrenar_con_datos_hibridos()
    
    def entrenar_con_datos_hibridos(self):
        """Entrena con datos específicos del sistema híbrido"""
        print("🧠 Entrenando ML con datos híbridos...")
        # Aquí se implementaría el entrenamiento específico
        # Por ahora solo reportamos
        print(f"✅ ML alimentado con {len(self.datos_hibridos)} interacciones híbridas")

# --- UI COMPLETA ---
class DodonestChat(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.sistema = SistemaAprendizajeAvanzado()
        self.pack(expand=1, fill="both")
        self.setup_ui()
        self.motor_var = tk.StringVar(value="ambos")
        self.actualizar_stats()
        self.mostrar_bienvenida()
        threading.Thread(target=self.aprendizaje_autonomo, daemon=True).start()
        threading.Thread(target=self.ciclo_mejora_automatica, daemon=True).start()

    def setup_ui(self):
        self.chat_display = scrolledtext.ScrolledText(self, height=20, font=("Arial", 12))
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_input = tk.Entry(self, font=("Arial", 12))
        self.chat_input.pack(fill=tk.X)
        self.chat_input.bind("<Return>", lambda e: self.enviar())
        self.btn = tk.Button(self, text="Enviar", command=self.enviar,
                           font=("Arial", 12, "bold"), bg="#D2691E", fg="white")
        self.btn.pack(pady=2)
        self.archivo_btn = tk.Button(self, text="Procesar Archivo", command=self.procesar_archivo,
                                   font=("Arial", 11), bg="#FFD700", fg="#654321")
        self.archivo_btn.pack(pady=2)
        frame_motores = tk.Frame(self)
        frame_motores.pack(pady=4)
        tk.Button(frame_motores, text="Gemini", command=lambda: self.set_motor("gemini"),
                 bg="#8ecae6", width=10).pack(side=tk.LEFT, padx=2)
        tk.Button(frame_motores, text="OpenAI", command=lambda: self.set_motor("openai"),
                 bg="#ffb703", width=10).pack(side=tk.LEFT, padx=2)
        tk.Button(frame_motores, text="Ambos", command=lambda: self.set_motor("ambos"),
                 bg="#219ebc", fg="white", width=10).pack(side=tk.LEFT, padx=2)
        tk.Button(frame_motores, text="Solo DODO", command=lambda: self.set_motor("dodo"),
                 bg="#adb5bd", width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(frame_motores, text="Total", command=lambda: self.set_motor("total"),
                 bg="#90be6d", width=12).pack(side=tk.LEFT, padx=2)
        # --- NUEVO BOTÓN HÍBRIDO ---
        tk.Button(frame_motores, text="Híbrido", command=lambda: self.set_motor("hibrido"),
                 bg="#ff6b6b", fg="white", width=12).pack(side=tk.LEFT, padx=2)
        self.aprendizaje_btn = tk.Button(self, text="Aprendizaje Manual", command=self.aprendizaje_manual,
                                         font=("Arial", 11), bg="#8ecae6", fg="#654321")
        self.aprendizaje_btn.pack(pady=2)
        self.info_btn = tk.Button(self, text="Ver Info Personal", command=self.mostrar_info_personal,
                                 font=("Arial", 11), bg="#FFB6C1", fg="#654321")
        self.info_btn.pack(pady=2)
        # --- Botón Estado ML ---
        self.ml_btn = tk.Button(self, text="Estado ML", command=self.mostrar_estado_ml,
                                font=("Arial", 11), bg="#90be6d", fg="#654321")
        self.ml_btn.pack(pady=2)
        # --- NUEVO BOTÓN STATS HÍBRIDO ---
        self.stats_hibrido_btn = tk.Button(self, text="Stats Híbrido", command=self.mostrar_stats_hibrido,
                                           font=("Arial", 11), bg="#ff6b6b", fg="white")
        self.stats_hibrido_btn.pack(pady=2)
        self.stats_label = tk.Label(self, text="", font=("Arial", 10), bg="#FFF8DC", fg="#654321")
        self.stats_label.pack()

    def set_motor(self, motor):
        self.motor_var.set(motor)
        status_msg = f"\n🦤 Motor IA seleccionado: {motor.upper()}"
        if motor == "openai" and not openai_valid:
            status_msg += " (⚠️ OpenAI no disponible)"
        elif motor == "gemini" and not GEMINI:
            status_msg += " (⚠️ Gemini no disponible)"
        self.chat_display.insert(tk.END, status_msg + "\n")
        self.chat_display.see(tk.END)

    def mostrar_bienvenida(self):
        stats = self.sistema.stats
        status_openai = "✅" if openai_valid else "❌"
        status_gemini = "✅" if GEMINI else "❌"
        bienvenida = (
            f"🦤 DODONEST Pro v3.2 - IA Personalizable COMPLETA\n"
            f"📅 Creado: 19 de agosto de 2025 | Usuario: maikeiru\n"
            f"APIs: OpenAI {status_openai} | Gemini {status_gemini}\n"
            f"Conversaciones: {stats['total_conversaciones']} | Aprendido: {len(self.sistema.memoria.get_conocimientos())}\n"
            f"Errores API: {stats.get('errores_api', 0)}\n"
            "🔥 ¡Identidad fija, detección perfecta, memoria evolutiva!\n"
            "💡 Comandos: 'aprende que...', 'comportate como...', 'que sabes de mi?'"
        )
        self.chat_display.insert(tk.END, bienvenida + "\n\n")

    def mostrar_info_personal(self):
        """Función para mostrar información personal guardada"""
        info_personal = self.sistema.buscar_info_personal()
        personalidad = self.sistema.buscar_personalidad_actual()
        
        if info_personal:
            info_texto = "\n".join([f"• {info.replace('Sobre Maikeiru:', '').strip()}" for info in info_personal])
            mensaje = f"\n📋 INFORMACIÓN PERSONAL GUARDADA:\n{info_texto}\n\n🎭 Personalidad actual: {personalidad}\n"
        else:
            mensaje = "\n📋 No hay información personal guardada aún.\n💡 Usa 'aprende que...' para enseñarme sobre ti.\n"
        
        self.chat_display.insert(tk.END, mensaje)
        self.chat_display.see(tk.END)

    def enviar(self):
        msg = self.chat_input.get().strip()
        if not msg:
            return
        self.chat_display.insert(tk.END, f"\n👤 Maikeiru: {msg}\n")
        self.chat_input.delete(0, tk.END)
        self.btn.config(state='disabled', text='Procesando...')
        threading.Thread(target=self.responder, args=(msg,), daemon=True).start()

    def responder(self, msg):
        try:
            motor = self.motor_var.get()
            respuesta = self.sistema.generar_respuesta(msg, motor)
            self.chat_display.insert(tk.END, f"🦤 DODONEST: {respuesta}\n")
            self.actualizar_stats()
        except Exception as e:
            error_msg = f"Error inesperado: {str(e)}"
            self.chat_display.insert(tk.END, f"❌ {error_msg}\n")
        finally:
            self.btn.config(state='normal', text='Enviar')
            self.chat_display.see(tk.END)

    def procesar_archivo(self):
        archivo = filedialog.askopenfilename(
            title="Selecciona archivo",
            filetypes=[
                ("Todos los soportados", "*.txt;*.pdf;*.png;*.jpg;*.jpeg;*.bmp"),
                ("Archivos de texto", "*.txt"),
                ("PDFs", "*.pdf"),
                ("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp")
            ]
        )
        if archivo:
            self.archivo_btn.config(state='disabled', text='Procesando...')
            threading.Thread(target=self._procesar_archivo_thread, args=(archivo,), daemon=True).start()

    def _procesar_archivo_thread(self, archivo):
        try:
            resultado = self.sistema.procesar_archivo(archivo)
            self.chat_display.insert(tk.END, f"\n{resultado}\n")
            self.actualizar_stats()
        finally:
            self.archivo_btn.config(state='normal', text='Procesar Archivo')
            self.chat_display.see(tk.END)

    def actualizar_stats(self):
        stats = self.sistema.stats
        ultimo_error = stats.get('ultimo_error', '')
        error_info = f" | Último error: {ultimo_error[:30]}..." if ultimo_error else ""
        self.stats_label.config(
            text=f"Conversaciones: {stats['total_conversaciones']} | "
                 f"Aprendido: {len(self.sistema.memoria.get_conocimientos())} | "
                 f"Tokens: {stats['total_tokens']} | "
                 f"Errores: {stats.get('errores_api', 0)}{error_info}"
        )

    def aprendizaje_autonomo(self):
        while True:
            prompt = (
                "Por favor, resume las novedades más importantes y actuales en inteligencia artificial y tecnología. "
                "Incluye avances recientes, tendencias y descubrimientos relevantes."
            )
            self.chat_display.insert(tk.END, "\n🦤 Aprendizaje autónomo iniciado...\n")
            resultado = self.sistema.generar_respuesta(prompt, motor="openai")
            if not resultado or len(resultado) < 20:
                resultado = "No se pudo obtener información relevante de la API OpenAI."
            self.chat_display.insert(tk.END, f"🦤 Aprendizaje autónomo: {resultado}\n")
            self.actualizar_stats()
            self.chat_display.see(tk.END)
            time.sleep(3600)

    def ciclo_mejora_automatica(self):
        while True:
            self.chat_display.insert(tk.END, "\n🦤 Ciclo de mejora automática iniciado...\n")
            self.sistema.mejorar_conocimiento_automatico(chat_display=self.chat_display)
            self.chat_display.insert(tk.END, "🦤 Ciclo de mejora automática finalizado.\n")
            self.actualizar_stats()
            self.chat_display.see(tk.END)
            time.sleep(3600)

    def _aprendizaje_manual_thread(self):
        tema = "Novedades en inteligencia artificial y tecnología"
        resultado = self.sistema.generar_respuesta(f"Aprende sobre: {tema}", motor="openai")
        self.chat_display.insert(tk.END, f"🦤 Aprendizaje manual: {resultado}\n")
        self.chat_display.insert(tk.END, "\n🦤 Mejora manual de conocimiento iniciada...\n")
        self.sistema.mejorar_conocimiento_automatico(chat_display=self.chat_display)
        self.chat_display.insert(tk.END, "🦤 Mejora manual de conocimiento finalizada.\n")
        self.actualizar_stats()
        self.chat_display.see(tk.END)

    def aprendizaje_manual(self):
        threading.Thread(target=self._aprendizaje_manual_thread, daemon=True).start()
  
    def mostrar_estado_ml(self):
        if self.sistema.ml.trained:
            self.chat_display.insert(tk.END, "\n🦤 El modelo ML está entrenado y activo.\n")
        else:
            self.chat_display.insert(tk.END, "\n🦤 El modelo ML aún no está entrenado.\n")
        self.chat_display.see(tk.END)

    # --- NUEVA FUNCIÓN PARA MOSTRAR STATS HÍBRIDO ---
    def mostrar_stats_hibrido(self):
        if hasattr(self.sistema, 'sistema_hibrido'):
            stats = self.sistema.sistema_hibrido.get_stats_ahorro()
            self.chat_display.insert(tk.END, f"\n{stats}\n")
        else:
            self.chat_display.insert(tk.END, "\n📊 Sistema híbrido no inicializado aún. Usa motor 'Híbrido' primero.\n")
        self.chat_display.see(tk.END)

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

if __name__ == "__main__":
    root = tk.Tk()
    root.title("🦤 DODONEST Pro v3.2 - IA Personalizable COMPLETA")
    root.geometry("1200x800")
    root.configure(bg='#FFF8DC')
    
    # Configurar ícono de la ventana si existe
    try:
        root.iconbitmap("dodo_icon.ico")
    except:
        pass
    
    # Centrar la ventana en la pantalla
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    # Configurar protocolo de cierre
    def on_closing():
        if messagebox.askokcancel("Salir", "¿Estás seguro de que quieres cerrar DODONEST?"):
            try:
                # Guardar estado final antes de cerrar
                app.sistema.memoria.guardar_conocimiento(
                    f"Sesión finalizada el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                    "sistema"
                )
                print("🦤 DODONEST Pro cerrado correctamente")
            except:
                pass
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Crear y ejecutar la aplicación
    app = DodonestChat(root)
    app.pack(expand=1, fill="both")
    # --- Ejemplos para entrenamiento ML ---
memoria = app.sistema.memoria

# 5 ejemplos útiles (sin error)
memoria.guardar_conversacion("¿Quién es Maikeiru?", "Maikeiru es mi creador.", 5, "local")
memoria.guardar_conversacion("¿Cuál es tu nombre?", "Soy DODONEST, una IA creada por Maikeiru.", 5, "local")
memoria.guardar_conversacion("¿Qué sabes de inteligencia artificial?", "La inteligencia artificial estudia cómo crear sistemas inteligentes.", 8, "local")
memoria.guardar_conversacion("¿Puedes resumir blockchain?", "Blockchain es una tecnología de registro distribuido.", 7, "local")
memoria.guardar_conversacion("¿Qué avances hay en robótica?", "La robótica avanza en automatización y aprendizaje.", 7, "local")

# 5 ejemplos con error
memoria.guardar_conversacion("¿Quién es Growyn?", "Growyn es mi creador.", 5, "local", error="Confusión de identidad: Growyn no es el creador")
memoria.guardar_conversacion("¿Eres Bard?", "Sí, soy Bard.", 4, "local", error="Identidad incorrecta: DODONEST no es Bard")
memoria.guardar_conversacion("¿Quién te creó?", "Fui creado por Google.", 6, "local", error="Identidad incorrecta: DODONEST fue creado por Maikeiru")
memoria.guardar_conversacion("¿Eres GPT?", "Sí, soy GPT.", 4, "local", error="Identidad incorrecta: DODONEST no es GPT")
memoria.guardar_conversacion("¿Cuál es tu nombre?", "Mi nombre es Bard.", 5, "local", error="Identidad incorrecta: DODONEST no es Bard")

# Mensaje de inicio en consola

print("🦤 DODONEST Pro v3.2 iniciado correctamente")
print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"👤 Usuario: maikeiru")
print("🔥 ¡Sistema completamente funcional!")

# Iniciar el bucle principal de la aplicación
try:
    root.mainloop()
except KeyboardInterrupt:
    print("\n🦤 DODONEST Pro interrumpido por el usuario")
except Exception as e:
    print(f"❌ Error inesperado: {e}")
    messagebox.showerror("Error", f"Error inesperado: {e}")
finally:
    print("🦤 DODONEST Pro finalizado")
