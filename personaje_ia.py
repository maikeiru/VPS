from memoria import MemoriaDodo

class PersonajeIA:
    """
    Clase base para un personaje IA con memoria persistente, estados y perfil.
    Puede usarse en videojuegos, simuladores o sistemas multi-agente.
    """

    def __init__(self, nombre, rol="NPC", personalidad=None, data_dir="personajes_memoria", db_file=None):
        self.nombre = nombre
        self.rol = rol
        self.personalidad = personalidad or "neutro"
        db_file = db_file or f"{nombre}_memoria.db"
        self.memoria = MemoriaDodo(data_dir=data_dir, db_file=db_file)
        self.estado = "inactivo"
        self.estado_conversacional = {}
        self.relaciones = {}  # {"nombre": {"tipo": "amigo/enemigo", "nivel": 5}}
        self.historial_chat = []

    def set_estado(self, nuevo_estado):
        self.estado = nuevo_estado

    def get_estado(self):
        return self.estado

    def set_estado_conversacional(self, clave, valor):
        self.estado_conversacional[clave] = valor

    def get_estado_conversacional(self, clave):
        return self.estado_conversacional.get(clave)

    def agregar_relacion(self, otro_personaje, tipo="amigo", nivel=5):
        self.relaciones[otro_personaje] = {"tipo": tipo, "nivel": nivel}

    def modificar_relacion(self, otro_personaje, tipo=None, nivel=None):
        if otro_personaje in self.relaciones:
            if tipo:
                self.relaciones[otro_personaje]["tipo"] = tipo
            if nivel is not None:
                self.relaciones[otro_personaje]["nivel"] = nivel

    def recordar_evento(self, evento, fuente="evento"):
        self.memoria.guardar_conocimiento(f"{self.nombre}: {evento}", fuente)

    def registrar_conversacion(self, pregunta, respuesta):
        self.historial_chat.append({"pregunta": pregunta, "respuesta": respuesta})
        self.memoria.guardar_conversacion(pregunta, respuesta, tokens=0, api="PersonajeIA", error=None)
        if len(self.historial_chat) > 10:
            self.historial_chat.pop(0)

    def generar_respuesta(self, mensaje, motor_func):
        """
        motor_func: función estilo llama3_query o similar.
        Enriquecer prompt con personalidad, estado, relaciones y memoria.
        """
        conocimientos = self.memoria.get_conocimientos(5)
        historial = "\n".join([f"Usuario: {item['pregunta']}\n{self.nombre}: {item['respuesta']}" for item in self.historial_chat])
        prompt = (
            f"Eres {self.nombre}, un personaje IA con rol {self.rol} y personalidad {self.personalidad}.\n"
            f"Estado actual: {self.estado}\n"
            f"Relaciones: {self.relaciones}\n"
            f"Historial reciente:\n{historial}\n"
            f"Conocimientos relevantes:\n{conocimientos}\n"
            f"\nNueva pregunta: {mensaje}\nResponde como {self.nombre}, usando tu personalidad y contexto."
        )
        respuesta = motor_func(prompt)
        self.registrar_conversacion(mensaje, respuesta)
        return respuesta