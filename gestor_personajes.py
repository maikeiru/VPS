import json
import os
from personaje_ia import PersonajeIA

class GestorPersonajes:
    def __init__(self, archivo_personajes="personajes.json"):
        self.archivo_personajes = archivo_personajes
        self.personajes = {}  # {"nombre": PersonajeIA}
        self.cargar_personajes()

    def agregar_personaje(self, nombre, rol="NPC", personalidad=None, estado="inactivo"):
        if nombre in self.personajes:
            print(f"⚠️ El personaje '{nombre}' ya existe.")
            return False
        personaje = PersonajeIA(nombre, rol, personalidad)
        personaje.set_estado(estado)
        self.personajes[nombre] = personaje
        self.guardar_personajes()
        return True

    def eliminar_personaje(self, nombre):
        if nombre in self.personajes:
            del self.personajes[nombre]
            self.guardar_personajes()
            return True
        print(f"⚠️ El personaje '{nombre}' no existe.")
        return False

    def obtener_personaje(self, nombre):
        return self.personajes.get(nombre)

    def listar_personajes(self):
        return list(self.personajes.keys())

    def exportar_personaje(self, nombre, archivo_export=None):
        personaje = self.personajes.get(nombre)
        if not personaje:
            print(f"⚠️ No existe el personaje '{nombre}'.")
            return
        datos = {
            "nombre": personaje.nombre,
            "rol": personaje.rol,
            "personalidad": personaje.personalidad,
            "estado": personaje.estado,
            "estado_conversacional": personaje.estado_conversacional,
            "relaciones": personaje.relaciones,
        }
        archivo_export = archivo_export or f"{nombre}_perfil.json"
        with open(archivo_export, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)
        print(f"Perfil de '{nombre}' exportado a '{archivo_export}'.")

    def guardar_personajes(self):
        datos = []
        for nombre, personaje in self.personajes.items():
            datos.append({
                "nombre": personaje.nombre,
                "rol": personaje.rol,
                "personalidad": personaje.personalidad,
                "estado": personaje.estado,
                "estado_conversacional": personaje.estado_conversacional,
                "relaciones": personaje.relaciones,
            })
        with open(self.archivo_personajes, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)

    def cargar_personajes(self):
        if not os.path.exists(self.archivo_personajes):
            return
        with open(self.archivo_personajes, "r", encoding="utf-8") as f:
            datos = json.load(f)
        for info in datos:
            personaje = PersonajeIA(
                info["nombre"],
                rol=info.get("rol", "NPC"),
                personalidad=info.get("personalidad", "neutro")
            )
            personaje.set_estado(info.get("estado", "inactivo"))
            personaje.estado_conversacional = info.get("estado_conversacional", {})
            personaje.relaciones = info.get("relaciones", {})
            self.personajes[info["nombre"]] = personaje