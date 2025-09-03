import os
import time
import re
from datetime import datetime
from memoria import MemoriaDodo
from ml import DodoML
from llama3_api import llama3_query
from construir_prompt import construir_prompt

# --------- CACHE DE RESPUESTAS ---------
class CacheRespuestas:
    def __init__(self, maxsize=1000):
        self.cache = {}
        self.maxsize = maxsize

    def get(self, pregunta):
        return self.cache.get(pregunta)

    def set(self, pregunta, respuesta):
        if len(self.cache) >= self.maxsize:
            # Elimina el primer elemento insertado (FIFO)
            self.cache.pop(next(iter(self.cache)))
        self.cache[pregunta] = respuesta

    def clear(self):
        self.cache = {}
# --------- FIN CACHE ---------

class SistemaAprendizajeAvanzado:
    def __init__(
        self,
        openai_client=None,
        gemini_model=None,
        data_dir=".",
        db_file="dodonest_memoria.db",
        json_file="dodonest_memoria.json",
        cache_respuestas=None
    ):
        self.memoria = MemoriaDodo(data_dir=data_dir, db_file=db_file, json_file=json_file)
        self.stats = self.memoria.get_stats()
        self.ml = DodoML(self.memoria)
        self.historial_chat = []
        self.rate_limit_delay = 1
        self.last_api_call = 0
        self.init_identidad()
        self.openai_client = openai_client
        self.gemini_model = gemini_model
        # Nueva cache de respuestas
        self.cache_respuestas = cache_respuestas if cache_respuestas is not None else CacheRespuestas(maxsize=1000)

    def init_identidad(self):
        conocimientos_identidad = [
            "Soy DODONEST, una IA creada por Maikeiru el 19 de agosto de 2025",
            "Mi creador es Maikeiru (maikeiru), quien programó mi código",
            "NUNCA soy Bard, GPT, Claude o cualquier otra IA - solo soy DODONEST",
            "Tengo memoria persistente y aprendo de cada conversación",
            "Uso APIs de OpenAI y Gemini para mejorar mis respuestas",
            "Mi propósito es ser útil, inteligente y mantener mi identidad",
            "Maikeiru es mi creador y puede enseñarme sobre él o modificar mi comportamiento"
        ]
        existentes = self.memoria.get_conocimientos(100)
        for conocimiento in conocimientos_identidad:
            if not any(conocimiento in c for c in existentes):
                self.memoria.guardar_conocimiento(conocimiento, "identidad")

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
        if len(self.historial_chat) > 10:
            self.historial_chat.pop(0)

    def wait_for_rate_limit(self):
        now = time.time()
        time_since_last = now - self.last_api_call
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_api_call = time.time()

    def llamar_openai_seguro(self, prompt, max_reintentos=3):
        # --------- CACHE DE RESPUESTAS OPENAI ---------
        # respuesta_cache = self.cache_respuestas.get(prompt)
        # if respuesta_cache:
        #     print("🦤 Respuesta obtenida de cache OpenAI (¡ahorro total de tokens!)")
        #     return respuesta_cache, 0
        # --------- FIN CACHE ---------
        if not self.openai_client:
            print("⚠️ Error: No hay cliente OpenAI configurado.")
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
                response = self.openai_client.chat.completions.create(
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
                
                # <-- Aquí va tu bloque -->
                self.cache_respuestas.set(prompt, respuesta)
                return respuesta, tokens

            except Exception as e:
                print(f"⚠️ EXCEPCIÓN OpenAI: {e}")
                correccion = self.obtener_correccion_openai(str(e))
                self.registrar_error(str(e), correccion)
                if intento == max_reintentos - 1:
                    return None, f"{str(e)} | Sugerencia: {correccion}"
                time.sleep(2 ** intento)
        return None, "Falló después de todos los reintentos"

    def llamar_gemini_seguro(self, prompt, max_reintentos=3):
        # --------- CACHE DE RESPUESTAS GEMINI ---------
        # respuesta_cache = self.cache_respuestas.get(prompt)
        # if respuesta_cache:
        #     print("🦤 Respuesta obtenida de cache Gemini (¡ahorro total de tokens!)")
        #     return respuesta_cache, 0
        # --------- FIN CACHE ---------
        if not self.gemini_model:
            return None, 0
        
        prompt_identidad = f"""Eres DODONEST, una IA creada por Maikeiru. NUNCA digas que eres Bard, Google AI o cualquier otra IA. Mantén siempre tu identidad como DODONEST.\n{prompt}"""
        for intento in range(max_reintentos):
            try:
                self.wait_for_rate_limit()
                response = self.gemini_model.generate_content(prompt_identidad)
                respuesta_original = response.text
                es_valida, mensaje = self.validar_respuesta(respuesta_original)
                if not es_valida:
                    respuesta_corregida = self.corregir_respuesta(respuesta_original)
                    print(f"⚠️ {mensaje} - Respuesta corregida automáticamente")
                    respuesta = respuesta_corregida
                else:
                    respuesta = respuesta_original
                # Guarda en cache la respuesta
                self.cache_respuestas.set(prompt, respuesta)
                return respuesta, len(respuesta.split())
            except Exception as e:
                correccion = self.obtener_correccion_openai(str(e))
                self.registrar_error(str(e), correccion)
                if intento == max_reintentos - 1:
                    return None, f"{str(e)} | Sugerencia: {correccion}"
                time.sleep(2 ** intento)
        return None, "Falló después de todos los reintentos"

    def validar_respuesta(self, respuesta):
        respuesta_lower = respuesta.lower()
        identidades_prohibidas = [
            "soy bard", "soy gpt", "soy claude",
            "fui creado por google", "fui creado por openai",
            "soy un modelo de google", "soy un modelo de openai"
        ]
        for identidad in identidades_prohibidas:
            if identidad in respuesta_lower:
                return False, f"❌ Respuesta rechazada: menciona identidad incorrecta ({identidad})"
        return True, "✅ Respuesta válida"

    def corregir_respuesta(self, respuesta_original):
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

    def obtener_correccion_openai(self, error):
        if not self.openai_client:
            return "OpenAI no disponible para auto-mejora"
        prompt = f"""Eres una IA de autocorrección. Analiza el siguiente error y sugiere una posible causa y corrección técnica para evitarlo en el futuro.
Error detectado:
{error}
Responde en formato breve y técnico."""
        try:
            self.wait_for_rate_limit()
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"No se pudo obtener corrección: {e}"

    def construir_prompt(self, mensaje, memoria, historial_chat):
        conocimientos = memoria.get_conocimientos(10)
        historial = "\n".join(
            [f"Usuario: {item['pregunta']}\nDODONEST: {item['respuesta']}" for item in historial_chat]
        )
        contexto = (
            "Eres DODONEST, una IA creada por Maikeiru. "
            "Recuerda tu identidad, historial de conversación y conocimientos aprendidos.\n"
            "Historial reciente:\n" + historial +
            "\nConocimientos aprendidos:\n" + "\n".join(conocimientos) +
            f"\n\nNueva pregunta: {mensaje}\nResponde de forma personalizada y útil."
        )
        return contexto
    
    def generar_respuesta_llama3local(self, mensaje):
        from construir_prompt import construir_prompt  # Puedes poner el import arriba del archivo si prefieres
        prompt = construir_prompt(
            mensaje,
            self.memoria,
            self.historial_chat
        )
        # --------- CACHE DE RESPUESTAS LLAMA3 ---------
        respuesta_cache = self.cache_respuestas.get(prompt)
        if respuesta_cache:
            print("🦤 Respuesta obtenida de cache Llama3 (¡ahorro total de tokens!)")
            return respuesta_cache
        # --------- FIN CACHE ---------
        from llama3_api import llama3_query  # Puedes poner el import arriba del archivo si prefieres
        respuesta = llama3_query(prompt)
        respuesta = respuesta.strip()
        if not respuesta or len(respuesta.split()) < 5:
            respuesta = "No puedo generar una respuesta útil con el modelo local. Por favor, usa el motor OpenAI."
        # Guarda en cache la respuesta
        self.cache_respuestas.set(prompt, respuesta)
        return respuesta
    
    def get_stats(self):
        return self.stats