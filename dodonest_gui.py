import tkinter as tk
from tkinter import scrolledtext, simpledialog, messagebox
from hibrido import SistemaHibridoMejorado
from cache_respuesta import CacheInteligente
from sistema import SistemaAprendizajeAvanzado
from optimizador_prompts import OptimizadorPrompts
from llama3_api import llama3_query
from gestor_personajes import GestorPersonajes
from plugin_manager import PluginManager
from config import load_api_keys, get_openai_client, get_gemini_model

class DodonestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DODONEST Chat & Console - Optimizado")
        self.root.geometry("900x700")
        self.root.configure(bg="#eaf6fc")
        
        # Models and managers con optimización
        self.OPENAI_API_KEY, self.GEMINI_API_KEY = load_api_keys()
        self.openai_client = get_openai_client(self.OPENAI_API_KEY)
        self.gemini_model = get_gemini_model(self.GEMINI_API_KEY)
        self.cache = CacheInteligente(maxsize=500)
        self.optimizador = OptimizadorPrompts()
        self.sistema = SistemaAprendizajeAvanzado(
            openai_client=self.openai_client,
            gemini_model=self.gemini_model,
            cache_respuestas=self.cache
        )
        self.sistema_hibrido = SistemaHibridoMejorado(self.sistema, self.cache, self.optimizador)
        self.gestor = GestorPersonajes()
        self.plugin_manager = PluginManager()
        self.plugin_manager.cargar_plugins()
        self.personaje_activo = None
        self.historial = []
        self.idioma_actual = "es"

        # Widgets
        self.create_widgets()
        self.print_welcome()

    def create_widgets(self):
        self.frame_top = tk.Frame(self.root, bg="#eaf6fc")
        self.frame_top.pack(fill=tk.X, padx=10, pady=5)

        self.cmd_label = tk.Label(self.frame_top, text="Comandos:", bg="#eaf6fc")
        self.cmd_label.pack(side=tk.LEFT, padx=5)
        self.cmd_entry = tk.Entry(self.frame_top, width=50)
        self.cmd_entry.pack(side=tk.LEFT, padx=5)
        self.cmd_btn = tk.Button(self.frame_top, text="Ejecutar", command=self.process_command)
        self.cmd_btn.pack(side=tk.LEFT, padx=5)

        self.chat_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=28, width=96, font=("Consolas", 12))
        self.chat_area.pack(padx=10, pady=10)
        self.chat_area.config(state=tk.DISABLED)

        self.frame_bottom = tk.Frame(self.root, bg="#eaf6fc")
        self.frame_bottom.pack(fill=tk.X, padx=10, pady=5)
        self.msg_label = tk.Label(self.frame_bottom, text="Mensaje:", bg="#eaf6fc")
        self.msg_label.pack(side=tk.LEFT, padx=5)
        self.msg_entry = tk.Entry(self.frame_bottom, width=70)
        self.msg_entry.pack(side=tk.LEFT, padx=5)
        self.send_btn = tk.Button(self.frame_bottom, text="Enviar", command=self.send_message)
        self.send_btn.pack(side=tk.LEFT, padx=5)

        self.func_btn = tk.Button(self.frame_bottom, text="Funciones disponibles", command=self.show_functions)
        self.func_btn.pack(side=tk.RIGHT, padx=5)

    def print_welcome(self):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, "🦤 Bienvenido a DODONEST GUI Chat Optimizado!\n")
        self.chat_area.insert(tk.END, "Sistema con cache inteligente y optimización de tokens activado.\n")
        self.chat_area.insert(tk.END, "Escribe tu mensaje abajo o ejecuta comandos en la barra superior.\n")
        self.chat_area.config(state=tk.DISABLED)

    def append_chat(self, text, sender="DODONEST"):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, f"{sender}: {text}\n")
        self.chat_area.see(tk.END)
        self.chat_area.config(state=tk.DISABLED)

    def send_message(self):
        msg = self.msg_entry.get().strip()
        if not msg:
            return

        # --- INTENCIÓN DE BÚSQUEDA WEB ---
        from intencion import analizar_intencion

        if analizar_intencion(msg) == "websearch":
            resultado = self.plugin_manager.ejecutar_plugin("websearch_plugin", msg)
            self.append_chat(resultado, sender="DODONEST")
            self.historial.append({"usuario": "Usuario", "mensaje": msg, "respuesta": resultado})
            self.msg_entry.delete(0, tk.END)
            return

        self.append_chat(msg, sender="Usuario")
        respuesta = self.get_response(msg)
        self.append_chat(respuesta, sender="DODONEST")
        self.historial.append({"usuario": "Usuario", "mensaje": msg, "respuesta": respuesta})
        self.msg_entry.delete(0, tk.END)
    
    def get_response(self, mensaje):
        # Recupera contexto simple (últimos 3 de historial)
        contexto_reciente = "\n".join([f"Q:{h['mensaje']} A:{h['respuesta']}" for h in self.historial[-3:]])
        pregunta_final = mensaje + ("\nContexto reciente:\n" + contexto_reciente if contexto_reciente else "")
        
        # Cache inteligente check
        respuesta_cache, tipo_cache = self.cache.get(pregunta_final)
        if respuesta_cache:
            return f"(cache {tipo_cache}) {respuesta_cache}"
        
        # Personaje activo
        if self.personaje_activo:
            respuesta = self.personaje_activo.generar_respuesta(pregunta_final, llama3_query)
        else:
            respuesta = self.sistema_hibrido.generar_respuesta_optimizada(pregunta_final)
        
        self.cache.set(pregunta_final, respuesta)
        return respuesta

    def process_command(self):
        cmd = self.cmd_entry.get().strip()
        if not cmd:
            return
        cmd_lower = cmd.lower()
        
        # NUEVOS COMANDOS DE OPTIMIZACIÓN
        if cmd_lower == "dashboard ahorro":
            stats_text = self.sistema_hibrido.get_dashboard_ahorro()
            self.append_chat(stats_text, sender="Sistema")
        elif cmd_lower == "limpiar cache":
            self.cache.limpiar_cache_antiguo(dias=7)
            self.cache.guardar_stats()
            self.append_chat("🧹 Cache limpiado y estadísticas guardadas", sender="Sistema")
        elif cmd_lower == "stats cache":
            stats = self.cache.get_estadisticas_ahorro()
            stats_text = f"📊 Tokens ahorrados: {stats['total_tokens_ahorrados']:,.0f}\n💰 Ahorro estimado: ${stats['ahorro_estimado_usd']:.4f}\n🎯 Hit rate: {stats['hit_rate_porcentaje']:.1f}%"
            self.append_chat(stats_text, sender="Sistema")
        
        # Comandos principales
        elif cmd_lower == "funciones":
            self.show_functions()
        elif cmd_lower.startswith("idioma:"):
            idioma = cmd.split("idioma:")[1].strip()
            self.idioma_actual = idioma
            self.append_chat(f"Idioma cambiado a: {idioma}", sender="Sistema")
        elif cmd_lower == "listar personajes":
            lista = self.gestor.listar_personajes()
            self.append_chat(f"Personajes: {', '.join(lista) if lista else 'Ninguno'}", sender="Sistema")
        elif cmd_lower.startswith("personaje:"):
            nombre = cmd.split(":", 1)[1].strip()
            personaje = self.gestor.obtener_personaje(nombre)
            if personaje:
                self.personaje_activo = personaje
                self.append_chat(f"Ahora conversando con: {nombre}", sender="Sistema")
            else:
                self.append_chat(f"No existe el personaje {nombre}.", sender="Sistema")
        elif cmd_lower == "desactivar personaje":
            self.personaje_activo = None
            self.append_chat("Personaje desactivado, vuelves al modo estándar.", sender="Sistema")
        elif cmd_lower.startswith("crear personaje:"):
            try:
                datos = cmd.split("crear personaje:")[1].strip().split(",")
                nombre = datos[0].strip()
                rol = datos[1].strip() if len(datos) > 1 else "NPC"
                personalidad = datos[2].strip() if len(datos) > 2 else None
                ok = self.gestor.agregar_personaje(nombre, rol, personalidad)
                self.append_chat(f"Personaje '{nombre}' creado." if ok else "No se pudo crear.", sender="Sistema")
            except Exception as e:
                self.append_chat("Error al crear personaje. Formato: crear personaje: nombre, rol, personalidad", sender="Sistema")
        elif cmd_lower.startswith("eliminar personaje:"):
            nombre = cmd.split("eliminar personaje:")[1].strip()
            ok = self.gestor.eliminar_personaje(nombre)
            self.append_chat(f"Personaje '{nombre}' eliminado." if ok else "No se pudo eliminar.", sender="Sistema")
        elif cmd_lower.startswith("exportar personaje:"):
            nombre = cmd.split("exportar personaje:")[1].strip()
            self.gestor.exportar_personaje(nombre)
            self.append_chat(f"Personaje '{nombre}' exportado.", sender="Sistema")
        elif cmd_lower.startswith("agregar plugin:"):
            ruta_plugin = cmd.split("agregar plugin:")[1].strip()
            try:
                self.plugin_manager.agregar_plugin(ruta_plugin)
                self.append_chat(f"Plugin agregado: {ruta_plugin}", sender="Sistema")
            except Exception as e:
                self.append_chat(f"Error al agregar plugin: {e}", sender="Sistema")
        elif cmd_lower.startswith("quitar plugin:"):
            nombre_plugin = cmd.split("quitar plugin:")[1].strip()
            try:
                self.plugin_manager.quitar_plugin(nombre_plugin)
                self.append_chat(f"Plugin quitado: {nombre_plugin}", sender="Sistema")
            except Exception as e:
                self.append_chat(f"Error al quitar plugin: {e}", sender="Sistema")
        elif cmd_lower == "ayuda":
            self.show_functions()
        elif cmd_lower == "salir":
            self.root.quit()
        else:
            self.append_chat("Comando no reconocido. Usa 'funciones' para ver las opciones.", sender="Sistema")
        self.cmd_entry.delete(0, tk.END)

    def show_functions(self):
        funciones = [
            "- crear personaje: nombre, rol, personalidad",
            "- eliminar personaje: nombre",
            "- listar personajes",
            "- personaje: nombre (activar)",
            "- desactivar personaje (volver a modo clásico)",
            "- exportar personaje: nombre",
            "- ayuda / funciones (muestra esta información)",
            "- idioma: [es|en]",
            "- agregar plugin: ruta.py",
            "- quitar plugin: nombre",
            "- dashboard ahorro (estadísticas de optimización)",
            "- limpiar cache (limpiar cache antiguo)",
            "- stats cache (estadísticas rápidas)",
            "- salir",
            "- Puedes chatear normalmente en el área de mensajes"
        ]
        self.append_chat("\n".join(funciones), sender="Funciones")

if __name__ == "__main__":
    root = tk.Tk()
    app = DodonestGUI(root)
    root.mainloop()