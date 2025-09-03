from intencion import analizar_intencion

def construir_prompt(mensaje, memoria, historial_chat):
    conocimientos = memoria.get_conocimientos(5)
    historial = "\n".join(
        [f"Usuario: {item['pregunta']}\nDODONEST: {item['respuesta']}" for item in historial_chat]
    )

    tipo = analizar_intencion(mensaje)
    if tipo == "breve":
        instruccion = (
            "Responde ÚNICAMENTE con una sola frase, máximo 3-5 líneas. "
            "NO incluyas contexto, ejemplos ni explicaciones adicionales. "
            "NO agregues ningún saludo, presentación ni cierre. "
            "Si tu respuesta excede este límite, finalízala abruptamente después de 5 líneas."
        )
        contexto = (
            "Eres DODONEST, una IA creada por Maikeiru.\n"
            f"{instruccion}\n"
            f"Pregunta: {mensaje}\n"
        )
    else:
        if tipo == "extenso":
            instruccion = "Responde de forma detallada y profesional, analizando el tema a fondo y aportando contexto y ejemplos si es relevante."
        elif tipo == "ejemplo":
            instruccion = "Da solo un ejemplo concreto, sin explicación adicional."
        elif tipo == "pasos":
            instruccion = "Responde indicando los pasos para realizar la acción, de forma clara y ordenada."
        elif tipo == "comparacion":
            instruccion = "Compara los conceptos mencionados, explicando sus diferencias y similitudes de forma clara."
        elif tipo == "insatisfaccion":
            instruccion = "Antes de responder, pregunta al usuario: '¿Prefieres una explicación breve, con ejemplos, o paso a paso?' No respondas hasta que el usuario elija."
        else:
            instruccion = "Responde de forma personalizada y útil."

        contexto = (
            "Eres DODONEST, una IA creada por Maikeiru. "
            "Recuerda tu identidad, historial de conversación y conocimientos aprendidos.\n"
            f"{instruccion}\n"
            "Historial reciente:\n" + historial +
            "\nConocimientos aprendidos:\n" + "\n".join(conocimientos) +
            f"\n\nNueva pregunta: {mensaje}\nResponde de forma personalizada y útil."
        )
    return contexto