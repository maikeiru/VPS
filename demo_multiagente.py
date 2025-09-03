from personaje_ia import PersonajeIA
from llama3_api import llama3_query

# Crear dos personajes IA
alice = PersonajeIA(nombre="Alice", rol="Exploradora", personalidad="curiosa")
bob = PersonajeIA(nombre="Bob", rol="Guardián", personalidad="protector")

# Estado conversacional: Alice está buscando un tesoro
alice.set_estado("buscando_tesoro")
alice.set_estado_conversacional("tema", "tesoro perdido")

# Relaciones
alice.agregar_relacion("Bob", tipo="amigo", nivel=8)
bob.agregar_relacion("Alice", tipo="amigo", nivel=7)

# Alice recuerda un evento
alice.recordar_evento("Descubrí una pista sobre el tesoro en el bosque.")

# Simulación de conversación
pregunta = "¿Qué sabes del tesoro perdido?"
respuesta_alice = alice.generar_respuesta(pregunta, llama3_query)
print(f"Alice: {respuesta_alice}")

# Bob responde sobre proteger el bosque
pregunta_bob = "¿Qué harías si alguien intenta robar el tesoro?"
respuesta_bob = bob.generar_respuesta(pregunta_bob, llama3_query)
print(f"Bob: {respuesta_bob}")

# Modificar relación por evento
bob.modificar_relacion("Alice", nivel=9)