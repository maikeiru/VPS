import uuid
from config import load_api_keys, get_openai_client, get_gemini_model
from sistema import SistemaAprendizajeAvanzado
from hibrido import SistemaHibridoMejorado
from cache_respuesta import CacheInteligente  # ✅ AGREGADO
from optimizador_prompts import OptimizadorPrompts  # ✅ AGREGADO
from llama3_api import llama3_query
from gestor_personajes import GestorPersonajes
from datetime import datetime
import json
from plugin_manager import PluginManager
from flask import Flask, request, jsonify

# ------------------- AGREGADO PARA RESPUESTAS BREVES -------------------
from recorte_breve import recortar_respuesta_breve
from intencion import analizar_intencion
# ------------------------------------------------------------------------

# Para soporte de idiomas
import gettext
import locale
import difflib

def buscar_info_memoria_fuzzy(mensaje, conocimientos, threshold=0.55):
    """
    Busca datos aprendidos en la memoria que sean similares a la pregunta.
    Prioriza conocimientos personales (frases con 'mi', 'me', 'soy', etc).
    """
    if not conocimientos:
        return None
        
    mensaje_lower = mensaje.lower()
    personales = []
    generales = []
    
    for c in conocimientos:
        if not c or not isinstance(c, str):
            continue
            
        cl = c.lower()
        try:
            sim = difflib.SequenceMatcher(None, mensaje_lower, cl).ratio()
        except Exception:
            continue
            
        # Si la similitud supera el umbral, lo considera relevante
        if sim > threshold:
            # Filtra si el conocimiento es personal
            if any(w in cl for w in ["mi", "me", "soy", "nací", "favorito", "tengo", "nacido", "llamado", "gustos"]):
                personales.append(f"✓ {c} (Similitud: {sim:.2f})")
            else:
                generales.append(f"• {c} (Similitud: {sim:.2f})")
    
    if personales:
        return "Recuerdo estos datos personales sobre ti:\n" + "\n".join(personales)
    elif generales:
        return "Recuerdo estos datos generales:\n" + "\n".join(generales)
    return None

# --------- CACHE INTELIGENTE OPTIMIZADO ---------
cache = CacheInteligente(maxsize=500)
optimizador_prompts = OptimizadorPrompts()
# --------- FIN CACHE ---------

plugin_manager = PluginManager()
plugin_manager.cargar_plugins()

# Memoria global de la sesión
historial_global = []
id_sesion = str(uuid.uuid4())
sistema_narrativa = None

IDIOMAS_DISPONIBLES = ["es", "en"]
idioma_actual = "es"  # Español por defecto

def set_idioma(nuevo_idioma):
    global idioma_actual, _
    if nuevo_idioma in IDIOMAS_DISPONIBLES:
        idioma_actual = nuevo_idioma
        try:
            localedir = "locales"
            lang = gettext.translation("messages", localedir, languages=[idioma_actual], fallback=True)
            lang.install()
            _ = lang.gettext
            print(_("🦤 Idioma cambiado a: ") + idioma_actual)
        except Exception:
            _ = lambda x: x
            print("⚠️ No se pudo cargar traducción, usando texto original.")
    else:
        print(f"⚠️ Idioma no soportado: {nuevo_idioma}")

# Inicializar traducción
set_idioma(idioma_actual)

# --- MEMORIA DE LARGO PLAZO ---
def guardar_memoria_contexto(contexto):
    try:
        with open("memoria_contexto.json", "w", encoding="utf-8") as f:
            json.dump(contexto, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error guardando memoria: {e}")

def cargar_memoria_contexto():
    try:
        with open("memoria_contexto.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

# --- AUTOCURACIÓN AUTOMÁTICA ---
def autocuracion(historial_global, sistema_hibrido, mensaje):
    try:
        N = 3
        if len(historial_global) < N:
            return None
        ultimos = historial_global[-N:]
        if all(e.get("feedback") == "👎" for e in ultimos):
            print(_("🦤 ⚠️ Se detectaron 3 respuestas inútiles seguidas. Activando autocuración..."))
            nueva_respuesta = sistema_hibrido.generar_respuesta_hibrida(mensaje + _(" Corrige el error y responde mejor."))
            print(_("🦤 Respuesta re-generada y autocurada!"))
            return nueva_respuesta
    except Exception as e:
        print(f"Error en autocuración: {e}")
    return None

def guardar_historial_global():
    try:
        with open("historial_global.json", "w", encoding="utf-8") as f:
            json.dump(historial_global, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error guardando historial: {e}")

def mostrar_ayuda():
    print(_("🦤 COMANDOS DISPONIBLES"))
    print(_("- crear personaje: nombre, rol, personalidad"))
    print(_("- eliminar personaje: nombre"))
    print(_("- listar personajes"))
    print(_("- personaje: nombre (activar)"))
    print(_("- desactivar personaje (volver a modo clásico)"))
    print(_("- exportar personaje: nombre"))
    print(_("- exportar historial (guarda historial_global.json)"))
    print(_("- ayuda (muestra esta información)"))
    print(_("- 👍 / 👎 (evalúa la última respuesta)"))
    print(_("- stats feedback (ver estadísticas de utilidad)"))
    print(_("- explica razonamiento (explica por qué se dio la última respuesta)"))
    print(_("- entrena ml (entrena el clasificador ML con tu feedback)"))
    print(_("- analiza imagen: ruta (describe el contenido de una imagen)"))
    print(_("- agregar plugin: ruta.py (agrega plugin en caliente)"))
    print(_("- quitar plugin: nombre (quita plugin en caliente)"))
    print(_("- benchmark ml (evalúa precisión del clasificador ML)"))
    print(_("- autocuración automática (activada)"))
    print(_("- memoria de largo plazo (activada)"))
    print(_("- idioma: [es|en] (cambia idioma de la interfaz y respuestas)"))
    print(_("- dashboard ahorro (ver estadísticas de tokens ahorrados)"))
    print(_("- limpiar cache (limpiar cache antiguo)"))
    print(_("- stats cache (estadísticas rápidas de cache)"))
    print(_("- iniciar historia aventura/misterio/profesional"))
    print(_("- pausar historia | resumen historia | exportar historia"))
    print(_("- salir"))

def calcular_stats_feedback(historial_global):
    try:
        total = 0
        utiles = 0
        for entrada in historial_global:
            if "feedback" in entrada:
                total += 1
                if entrada["feedback"] == "👍":
                    utiles += 1
        if total == 0:
            return _("No hay feedback aún.")
        pct = utiles / total * 100
        return _("Respuestas útiles: {}/{} ({:.1f}%)").format(utiles, total, pct)
    except Exception as e:
        print(f"Error calculando stats: {e}")
        return "Error calculando estadísticas"

def entrenar_ml_incremental():
    try:
        from dodo_ml import DodoML
        X_train = []
        y_train = []
        for entrada in historial_global:
            if "feedback" in entrada:
                texto = f"{entrada.get('mensaje', '')} {entrada.get('respuesta', '')}"
                X_train.append(texto)
                y_train.append(1 if entrada["feedback"] == "👍" else 0)
        if not X_train:
            print(_("🦤 No hay datos suficientes para entrenar."))
            return
        modelo = DodoML()
        modelo.entrenar(X_train, y_train)
        modelo.guardar_modelo()
        print(_("🦤 Modelo ML entrenado con {} ejemplos.").format(len(X_train)))
    except Exception as e:
        print(_("🦤 Error entrenando ML: {}").format(e))
def procesar_mensaje(mensaje):
    """
    Procesa el mensaje usando el sistema híbrido y devuelve la respuesta.
    """
    try:
        # Inicializa dependencias si no existen (puedes adaptar según tu estructura)
        OPENAI_API_KEY, GEMINI_API_KEY = load_api_keys()
        openai_client = get_openai_client(OPENAI_API_KEY)
        gemini_model = get_gemini_model(GEMINI_API_KEY)

        sistema = SistemaAprendizajeAvanzado(
            openai_client=openai_client,
            gemini_model=gemini_model,
            cache_respuestas=cache
        )
        sistema_hibrido = SistemaHibridoMejorado(sistema, cache, optimizador_prompts)
        # Procesa el mensaje con el sistema híbrido
        respuesta = sistema_hibrido.generar_respuesta_optimizada(mensaje)
        return respuesta
    except Exception as e:
        return f"Error procesando mensaje: {e}"

def main():
    global sistema_narrativa
    
    try:
        OPENAI_API_KEY, GEMINI_API_KEY = load_api_keys()
        openai_client = get_openai_client(OPENAI_API_KEY)
        gemini_model = get_gemini_model(GEMINI_API_KEY)

        sistema = SistemaAprendizajeAvanzado(
            openai_client=openai_client,
            gemini_model=gemini_model,
            cache_respuestas=cache
        )
        
        # Crear el sistema híbrido con los parámetros correctos
        sistema_hibrido = SistemaHibridoMejorado(sistema, cache, optimizador_prompts)
        
        gestor = GestorPersonajes()
        personaje_activo = None

        print(_("🦤 DODONEST Console Test (Sesión: {})").format(id_sesion))
        mostrar_ayuda()

        while True:
            try:
                modo = _("(Personaje activo: {})").format(personaje_activo.nombre) if personaje_activo else _("(Modo estándar)")
                mensaje = input(_("\n💤 Maikeiru {}: ").format(modo))

                if not mensaje.strip():
                    continue

                # DEBUG: Mostrar comando detectado
                print(f"[DEBUG] Comando detectado: '{mensaje}'")

                # --- CAMBIO DE IDIOMA ---
                if mensaje.lower().startswith("idioma:"):
                    nuevo_idioma = mensaje.split("idioma:")[1].strip()
                    set_idioma(nuevo_idioma)
                    continue

                # --- NUEVOS COMANDOS DE OPTIMIZACIÓN ---
                if mensaje.lower().strip() == "dashboard ahorro":
                    try:
                        print(sistema_hibrido.get_dashboard_ahorro())
                    except Exception as e:
                        print(f"Error mostrando dashboard: {e}")
                    continue

                if mensaje.lower().strip() == "limpiar cache":
                    try:
                        cache.limpiar_cache_antiguo(dias=7)
                        cache.guardar_stats()
                        print(_("🧹 Cache limpiado y estadísticas guardadas"))
                    except Exception as e:
                        print(f"Error limpiando cache: {e}")
                    continue

                if mensaje.lower().strip() == "stats cache":
                    try:
                        stats = cache.get_estadisticas_ahorro()
                        print(f"📊 Tokens ahorrados: {stats['total_tokens_ahorrados']:,.0f}")
                        print(f"💰 Ahorro estimado: ${stats['ahorro_estimado_usd']:.4f}")
                        print(f"🎯 Hit rate: {stats['hit_rate_porcentaje']:.1f}%")
                    except Exception as e:
                        print(f"Error mostrando stats: {e}")
                    continue

                # --- SISTEMA DE NARRATIVA ---
                if mensaje.lower().startswith("iniciar historia"):
                    try:
                        tipo = "aventura"  # por defecto
                        if "misterio" in mensaje.lower():
                            tipo = "misterio"
                        elif "profesional" in mensaje.lower():
                            tipo = "profesional"
                        
                        sistema_narrativa = SistemaNarrativa(gestor)
                        sistema_narrativa.iniciar_historia(tipo)
                    except Exception as e:
                        print(f"Error iniciando historia: {e}")
                    continue

                if mensaje.lower().strip() == "pausar historia":
                    if sistema_narrativa:
                        sistema_narrativa.pausar_historia()
                    else:
                        print("⚠️ No hay historia activa")
                    continue

                if mensaje.lower().strip() == "resumen historia":
                    if sistema_narrativa:
                        sistema_narrativa.mostrar_resumen()
                    else:
                        print("⚠️ No hay historia activa")
                    continue

                if mensaje.lower().strip() == "exportar historia":
                    if sistema_narrativa:
                        sistema_narrativa.exportar_historia()
                    else:
                        print("⚠️ No hay historia activa")
                    continue

                # --- FEEDBACK HUMANO ---
                if mensaje.strip() in ["👍", "👎"]:
                    if len(historial_global) == 0:
                        print(_("⚠️ No hay respuesta previa para evaluar."))
                        continue
                    historial_global[-1]["feedback"] = mensaje.strip()
                    guardar_historial_global()
                    print(_("🦤 Feedback registrado: {}").format(mensaje.strip()))
                    print(_("🦤 Estadísticas de utilidad:"), calcular_stats_feedback(historial_global))

                    # --- AUTOCURACIÓN AUTOMÁTICA ---
                    if mensaje.strip() == "👎":
                        nueva = autocuracion(historial_global, sistema_hibrido, historial_global[-1]["mensaje"])
                        if nueva:
                            historial_global[-1]["respuesta"] = nueva
                            print(_("🦤 Nueva respuesta autocurada: {}").format(nueva))
                    # --- ENTRENAMIENTO INCREMENTAL DEL ML ---
                    entrenar_ml_incremental()
                    continue

                # --- VER ESTADÍSTICAS DE FEEDBACK ---
                if mensaje.lower().strip() == "stats feedback":
                    print(_("🦤 Estadísticas de utilidad:"), calcular_stats_feedback(historial_global))
                    continue

                # --- EXPLICABILIDAD / TRANSPARENCIA ---
                if mensaje.lower().strip() == "explica razonamiento":
                    if len(historial_global) == 0:
                        print(_("🦤 No hay respuesta previa para explicar."))
                        continue
                    ultima = historial_global[-1]
                    print(_("🦤 Explicación razonamiento:"))
                    print(_("- Pregunta recibida: {}").format(ultima.get('mensaje')))
                    print(_("- Respuesta dada: {}").format(ultima.get('respuesta')))
                    print(_("- Método/modelo usado: {}").format('PersonajeIA' if ultima.get('personaje_activo') else 'DODONEST sistema híbrido'))
                    print(_("- Stats: {}").format(ultima.get('stats')))
                    continue

                # --- ENTRENAMIENTO MANUAL DE CLASIFICADOR ML ---
                if mensaje.lower().strip() == "entrena ml":
                    entrenar_ml_incremental()
                    continue

                # --- BENCHMARK AUTOMÁTICO ---
                if mensaje.lower().strip() == "benchmark ml":
                    try:
                        from dodo_ml import DodoML
                        X_test = []
                        y_test = []
                        for entrada in historial_global:
                            if "feedback" in entrada:
                                texto = f"{entrada.get('mensaje', '')} {entrada.get('respuesta', '')}"
                                X_test.append(texto)
                                y_test.append(1 if entrada["feedback"] == "👍" else 0)
                        if not X_test:
                            print(_("🦤 No hay datos de feedback para benchmark."))
                            continue
                        modelo = DodoML()
                        y_pred = modelo.predecir(X_test)
                        aciertos = sum([1 for yp, yt in zip(y_pred, y_test) if yp == yt])
                        print(_("🦤 Precisión del clasificador ML en feedback: {}/{} ({:.1f}%)").format(aciertos, len(y_test), aciertos/len(y_test)*100))
                    except Exception as e:
                        print(_("🦤 Error en benchmark ML: {}").format(e))
                    continue

                # --- ANÁLISIS DE IMAGENES ---
                if mensaje.lower().startswith("analiza imagen:"):
                    ruta_imagen = mensaje.split("analiza imagen:")[1].strip()
                    try:
                        resultado = plugin_manager.ejecutar_plugin("vision_plugin", ruta_imagen)
                        print(_("🦤 Análisis visual:\n{}").format(resultado))
                    except Exception as e:
                        print(f"Error analizando imagen: {e}")
                    continue

                # --- GESTIÓN DINÁMICA DE PLUGINS ---
                if mensaje.lower().startswith("agregar plugin:"):
                    ruta_plugin = mensaje.split("agregar plugin:")[1].strip()
                    try:
                        plugin_manager.agregar_plugin(ruta_plugin)
                        print(_("🦤 Plugin agregado: {}").format(ruta_plugin))
                    except Exception as e:
                        print(_("⚠️ Error al agregar plugin: {}").format(e))
                    continue

                if mensaje.lower().startswith("quitar plugin:"):
                    nombre_plugin = mensaje.split("quitar plugin:")[1].strip()
                    try:
                        plugin_manager.quitar_plugin(nombre_plugin)
                        print(_("🦤 Plugin quitado: {}").format(nombre_plugin))
                    except Exception as e:
                        print(_("⚠️ Error al quitar plugin: {}").format(e))
                    continue

                if mensaje.strip().lower() in ["salir", "exit", "quit"]:
                    print(_("🦤 Saliendo de DODONEST..."))
                    guardar_historial_global()
                    break

                if mensaje.lower().startswith("ayuda"):
                    mostrar_ayuda()
                    continue

                if mensaje.lower().startswith("desactivar personaje"):
                    personaje_activo = None
                    print(_("🦤 Personaje desactivado, vuelves al modo estándar."))
                    continue

                if mensaje.lower().startswith("crear personaje:"):
                    try:
                        datos_raw = mensaje.split("crear personaje:", 1)[1].strip()
                        
                        # Si no hay comas, asumir que solo es el nombre
                        if "," not in datos_raw:
                            nombre = datos_raw.strip()
                            rol = "NPC"
                            personalidad = "amigable"
                            descripcion = ""
                        else:
                            # Parsear con comas
                            partes = [d.strip() for d in datos_raw.split(",")]
                            nombre = partes[0] if len(partes) > 0 else "Personaje"
                            rol = partes[1] if len(partes) > 1 else "NPC"
                            personalidad = partes[2] if len(partes) > 2 else "amigable"
                            descripcion = partes[3] if len(partes) > 3 else ""
                        
                        if not nombre:
                            print(_("⚠️ Error: Debes especificar al menos el nombre del personaje."))
                            print("Ejemplos:")
                            print("crear personaje: Dr.Martinez, doctor, profesional, Médico con 20 años de experiencia")
                            print("crear personaje: Dr.Martinez")
                            continue
                        
                        ok = gestor.agregar_personaje(nombre, rol, personalidad, descripcion)
                        if ok:
                            print(_("🦤 Personaje '{}' creado exitosamente.").format(nombre))
                            print(f"   Rol: {rol}")
                            print(f"   Personalidad: {personalidad}")
                            if descripcion:
                                print(f"   Descripción: {descripcion}")
                        else:
                            print(_("⚠️ No se pudo crear el personaje. Puede que ya exista."))
                            
                    except Exception as e:
                        print(_("⚠️ Error al crear personaje. Usa el formato:"))
                        print("crear personaje: nombre, rol, personalidad, descripción")
                        print("O simplemente: crear personaje: nombre")
                        print(f"DEBUG: {e}")
                    continue

                if mensaje.lower().startswith("eliminar personaje:"):
                    nombre = mensaje.split("eliminar personaje:")[1].strip()
                    try:
                        ok = gestor.eliminar_personaje(nombre)
                        print(_("🦤 Personaje '{}' eliminado.").format(nombre) if ok else _("⚠️ No se pudo eliminar."))
                    except Exception as e:
                        print(f"Error eliminando personaje: {e}")
                    continue

                if mensaje.lower().strip() == "listar personajes":
                    try:
                        lista = gestor.listar_personajes()
                        print(_("🦤 Personajes disponibles: {}").format(', '.join(lista) if lista else _("Ninguno")))
                    except Exception as e:
                        print(f"Error listando personajes: {e}")
                    continue

                if mensaje.lower().startswith("personaje:"):
                    nombre = mensaje.split(":", 1)[1].strip()
                    try:
                        personaje = gestor.obtener_personaje(nombre)
                        if personaje:
                            personaje_activo = personaje
                            print(_("🦤 Ahora conversando con el personaje: {}").format(personaje_activo.nombre))
                        else:
                            print(_("🦤 No existe el personaje '{}'. Usa 'listar personajes' para ver los existentes.").format(nombre))
                    except Exception as e:
                        print(f"Error activando personaje: {e}")
                    continue

                if mensaje.lower().startswith("resumen"):
                    try:
                        resultado = plugin_manager.ejecutar_plugin("resumen_plugin", historial_global)
                        print(_("🦤 Resumen de la sesión:\n{}").format(resultado))
                    except Exception as e:
                        print(f"Error generando resumen: {e}")
                    continue

                if mensaje.lower().startswith("exportar personaje:"):
                    nombre = mensaje.split("exportar personaje:")[1].strip()
                    try:
                        gestor.exportar_personaje(nombre)
                    except Exception as e:
                        print(f"Error exportando personaje: {e}")
                    continue

                if mensaje.lower().startswith("exportar historial"):
                    guardar_historial_global()
                    print(_("🦤 Historial global exportado a historial_global.json"))
                    continue

                if analizar_intencion(mensaje) == "websearch":
                    try:
                        resultado = plugin_manager.ejecutar_plugin("websearch_plugin", mensaje)
                        print(_("🦤 Resultado búsqueda web:\n{}").format(resultado))
                    except Exception as e:
                        print(f"Error en búsqueda web: {e}")
                    continue

                # --- Consulta general de memoria antes de generar la respuesta ---
                try:
                    conocimientos = sistema.memoria.get_conocimientos(50)
                    respuesta_memoria = buscar_info_memoria_fuzzy(mensaje, conocimientos)
                    if respuesta_memoria:
                        print(_("🦤 DODONEST: {}").format(respuesta_memoria))
                        entrada = {
                            "usuario": "Maikeiru",
                            "mensaje": mensaje,
                            "personaje_activo": personaje_activo.nombre if personaje_activo else None,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "id_sesion": id_sesion,
                            "respuesta": respuesta_memoria,
                            "stats": sistema.get_stats(),
                            "metodo_usado": "consulta_memoria_fuzzy"
                        }
                        historial_global.append(entrada)
                        continue
                except Exception as e:
                    print(f"Error en consulta memoria: {e}")

                # Procesamiento de conversación
                tiempo = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                entrada = {
                    "usuario": "Maikeiru",
                    "mensaje": mensaje,
                    "personaje_activo": personaje_activo.nombre if personaje_activo else None,
                    "timestamp": tiempo,
                    "id_sesion": id_sesion
                }

                # --- Recupera contexto útil ---
                contexto = cargar_memoria_contexto()
                contexto_reciente = "\n".join([_("Q:{} A:{}").format(c['pregunta'], c['respuesta']) for c in contexto[-3:]]) if contexto else ""
                
                # SEPARAR: pregunta_final para cache/APIs vs mensaje_limpio para personajes
                pregunta_final = mensaje
                if contexto_reciente: 
                    pregunta_final += _("\nContexto reciente:\n") + contexto_reciente
                
                mensaje_limpio = mensaje  # 🔥 NUEVO: mensaje sin contaminar para personajes

                respuesta = None
                metodo_usado = None
                tokens_usados = None

                # ---------- CACHE INTELIGENTE MEJORADO ----------
                try:
                    respuesta_cache, tipo_cache = cache.get(pregunta_final)
                    if respuesta_cache and isinstance(respuesta_cache, str) and len(respuesta_cache.strip()) > 5:
                        print(_("🦤 Respuesta obtenida de cache inteligente (¡ahorro total de tokens!)"))
                        respuesta = respuesta_cache
                        tokens_usados = 0
                        metodo_usado = f"cache_{tipo_cache}"
                    else:
                        # ----------- Conversación normal -----------
                        if personaje_activo:
                            # 🔥 FIX: usar mensaje_limpio en lugar de pregunta_final
                            respuesta = personaje_activo.generar_respuesta(mensaje_limpio, llama3_query)
                            metodo_usado = "PersonajeIA"
                            tokens_usados = 0
                        else:
                            resultado = sistema_hibrido.generar_respuesta_optimizada(pregunta_final)
                            
                            # Manejar correctamente tuplas/strings
                            if isinstance(resultado, tuple) and len(resultado) == 2:
                                respuesta, tokens_usados = resultado
                            elif isinstance(resultado, str):
                                respuesta = resultado
                                tokens_usados = 0  # Llama3 local
                            else:
                                print("⚠️ Error en respuesta del sistema, usando respaldo")
                                respuesta = "Como DODONEST, necesito más información para responder adecuadamente."
                                tokens_usados = 0
                            
                            metodo_usado = "hibrido_optimizado"

                        # Guarda la nueva respuesta en cache SOLO si es válida
                        if isinstance(respuesta, str) and len(respuesta.strip()) > 10 and "¿en qué puedo ayudarte?" not in respuesta.lower():
                            cache.set(pregunta_final, respuesta)
                except Exception as e:
                    print(f"Error en cache: {e}")
                    respuesta = "Error procesando respuesta"
                    tokens_usados = 0
                    metodo_usado = "error"

                # ---------- FIN CACHE INTELIGENTE ----------

                # -------------------- AGREGADO BREVES ---------------------
                # Aplica el recorte de respuesta si la intención es "breve"
                try:
                    if analizar_intencion(mensaje) == "breve":
                        respuesta = recortar_respuesta_breve(respuesta)
                except Exception as e:
                    print(f"Error recortando respuesta: {e}")
                # ----------------------------------------------------------

                # --- PREDICCIÓN DE UTILIDAD ANTES DE MOSTRAR RESPUESTA ---
                try:
                    from dodo_ml import DodoML
                    modelo = DodoML()
                    if modelo.modelo_cargado():
                        texto_pred = f"{mensaje} {respuesta}"
                        utilidad_predicha = modelo.predecir([texto_pred])[0]
                        if utilidad_predicha == 0:
                            print(_("🦤 ⚠️ La respuesta predicha sería poco útil. Intentando mejorar..."))
                            respuesta_mejorada = sistema_hibrido.generar_respuesta_hibrida(mensaje + _(" Sé más claro y útil."))
                            if respuesta_mejorada and respuesta_mejorada != "None":
                                respuesta = respuesta_mejorada
                                print(_("🦤 Respuesta mejorada generada por el sistema."))
                            else:
                                print(_("🦤 No se pudo mejorar la respuesta automáticamente."))
                except Exception as e:
                    print(_("🦤 (ML: {})").format(e))

                # Imprime la respuesta
                if personaje_activo:
                    print("{}: {}".format(personaje_activo.nombre, respuesta))
                else:
                    print(_("🦤 DODONEST: {}").format(respuesta))

                try:
                    stats = sistema.get_stats()
                    # Usar tokens de la conversación actual
                    if tokens_usados is not None and tokens_usados > 0:
                        tokens_display = tokens_usados
                    else:
                        tokens_display = "0 (Llama3 local)"
                        
                    print(_("Conversaciones: {} | Tokens: {} | Errores: {}").format(
                        stats.get('total_conversaciones', 0),
                        tokens_display,
                        stats.get('errores_api', 0)
                    ))
                except Exception as e:
                    print(f"Error mostrando stats: {e}")

                # Guarda en memoria global
                entrada["respuesta"] = respuesta
                entrada["stats"] = sistema.get_stats() if hasattr(sistema, 'get_stats') else {}
                entrada["metodo_usado"] = metodo_usado
                historial_global.append(entrada)

                # --- Actualiza memoria de contexto ---
                if respuesta and len(respuesta) > 30:
                    try:
                        contexto.append({
                            "pregunta": mensaje,
                            "respuesta": respuesta,
                            "timestamp": tiempo
                        })
                        guardar_memoria_contexto(contexto)
                    except Exception as e:
                        print(f"Error guardando contexto: {e}")

            except KeyboardInterrupt:
                print("\n" + _("🦤 Interrupción detectada. Saliendo..."))
                break
            except Exception as e:
                print(f"Error en bucle principal: {e}")
                continue

    except Exception as e:
        print(f"Error crítico en main: {e}")

if __name__ == "__main__":
    main()
