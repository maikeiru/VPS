import importlib.util
import importlib
import os
import sys

class PluginManager:
    def __init__(self, plugins_dir="plugins"):
        self.plugins_dir = plugins_dir
        self.plugins = {}

    def cargar_plugins(self):
        plugins_abspath = os.path.abspath(self.plugins_dir)
        if plugins_abspath not in sys.path:
            sys.path.insert(0, plugins_abspath)
        for archivo in os.listdir(self.plugins_dir):
            if archivo.endswith(".py") and archivo != "__init__.py":
                nombre = archivo[:-3]
                ruta = os.path.join(self.plugins_dir, archivo)
                self._cargar_plugin(ruta, nombre)

    def _cargar_plugin(self, ruta, nombre):
        try:
            spec = importlib.util.spec_from_file_location(nombre, ruta)
            modulo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(modulo)
            self.plugins[nombre] = modulo
            print(f"🦤 Plugin cargado: {nombre}")
        except Exception as e:
            print(f"⚠️ Error cargando plugin {nombre}: {e}")

    def ejecutar_plugin(self, nombre, *args, **kwargs):
        plugin = self.plugins.get(nombre)
        if not plugin:
            print(f"⚠️ Plugin '{nombre}' no está cargado.")
            return None
        # Busca método main, ejecutar, run, etc.
        for method_name in ["main", "ejecutar", "run"]:
            if hasattr(plugin, method_name):
                method = getattr(plugin, method_name)
                try:
                    return method(*args, **kwargs)
                except Exception as e:
                    print(f"⚠️ Error ejecutando plugin {nombre}: {e}")
                    return None
        print(f"⚠️ Plugin '{nombre}' no tiene método principal ('main', 'ejecutar' o 'run').")
        return None

    def agregar_plugin(self, ruta):
        """
        Agrega un plugin dinámicamente en caliente, desde ruta absoluta o relativa.
        """
        nombre = os.path.basename(ruta).replace(".py","")
        self._cargar_plugin(ruta, nombre)

    def quitar_plugin(self, nombre):
        """
        Quita un plugin de la lista de plugins cargados.
        """
        if nombre in self.plugins:
            del self.plugins[nombre]
            print(f"🦤 Plugin '{nombre}' quitado.")
        else:
            print(f"⚠️ No se encontró el plugin '{nombre}' para quitar.")
