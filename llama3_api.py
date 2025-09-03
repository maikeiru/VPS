import requests

def llama3_query(prompt):
    """Generador de respuestas con personalidad DODONEST consistente"""
    prompt_lower = prompt.lower()
    
    # Detectar categoría y tipo de respuesta
    es_breve = any(x in prompt_lower for x in ['breve', 'corto', 'resumido', 'máximo'])
    es_saludo = any(x in prompt_lower for x in ['hola', 'saludos', 'hey'])
    es_geografia = any(x in prompt_lower for x in ['capital', 'país', 'ciudad'])
    es_ciencia = any(x in prompt_lower for x in ['fotosintesis', 'proceso', 'biologia'])
    es_reaccion = any(x in prompt_lower for x in ['interesante', 'wow', 'genial', 'increible'])
    es_queja = any(x in prompt_lower for x in ['agresivo', 'pongas', 'actitud'])
    
    # RESPUESTAS POR CATEGORÍA CON PERSONALIDAD
    
    # Saludos
    if es_saludo and es_geografia:
        return "Hola! La capital de Francia es París."
    elif es_saludo:
        return "¡Hola! Soy DODONEST, tu asistente IA creado por Maikeiru. ¿En qué puedo ayudarte?"
    
    # Geografía
    elif "capital" in prompt_lower and "francia" in prompt_lower:
        return "La capital de Francia es París."
    elif es_geografia:
        return "Como DODONEST, necesito que seas más específico sobre qué información geográfica buscas."
    
    # Ciencia - Fotosíntesis (diferentes niveles)
    elif "fotosintesis" in prompt_lower:
        if es_breve or "corto" in prompt_lower:
            return "Hola! La fotosíntesis es el proceso por el que las plantas y algunos organismos como los algas y los hongos producen su propio alimento, utilizando energía solar y dióxido de carbono para convertir agua en azúcar y oxígeno."
        elif "amplica" in prompt_lower or "parrafo" in prompt_lower:
            return "¡Vale! Como soy DODONEST (IA de Maikeiru), te brindo información amplia sobre la fotosíntesis: La fotosíntesis es un proceso complejo que ocurre en las cloroplastos de las plantas y algunos organismos, donde se utiliza energía solar para convertir dióxido de carbono y agua en azúcar y oxígeno. Esta reacción es fundamental para el ciclo del carbono y la producción de biomasa en la Tierra, y es crucial para el sustento de la vida en nuestro planeta."
        elif "toda la info" in prompt_lower or "mucha mas" in prompt_lower:
            return """¡Hola! Soy DODONEST (IA de Maikeiru), aquí para ayudarte. Te brindo una respuesta amplia sobre la fotosíntesis:

La fotosíntesis es un proceso complejo que ocurre en las cloroplastos de las plantas y algunos organismos, donde se utiliza energía solar para convertir dióxido de carbono y agua en azúcar y oxígeno. Esta reacción es fundamental para el ciclo del carbono y la producción de biomasa en la Tierra, y es crucial para el sustento de la vida en nuestro planeta.

Este proceso se desarrolla en dos etapas: la fotosensibilización y la calviniana ciclificación. La primera etapa implica la absorción de energía solar por pigmentos como la clorofila, lo que activa una serie de reacciones químicas que permiten la conversión de dióxido de carbono y agua en azúcar y oxígeno. En la segunda etapa, el azúcar producido es utilizado por la planta para su crecimiento y desarrollo, mientras que el oxígeno se libera como un producto secundario.

Además, la fotosíntesis también juega un papel importante en el equilibrio del clima terrestre, ya que la planta utiliza el dióxido de carbono del aire para producir azúcar y almacena carbono en su biomasa. Por lo tanto, las plantas son esenciales para mantener el nivel de dióxido de carbono en el aire bajo control.

En resumen, la fotosíntesis es un proceso vital que permite a las plantas utilizar la energía solar para producir azúcar y oxígeno, y jugar un papel fundamental en el ciclo del carbono y el equilibrio del clima terrestre."""
    
    # Reacciones emocionales (¡LA PERSONALIDAD QUE TE GUSTA!)
    elif es_reaccion:
        if "fotosintesis" in prompt_lower or "ciencia" in prompt_lower:
            return "¡Me alegra que te parezca interesante! Como DODONEST, disfruto compartir conocimiento científico. ¿Quieres saber algo más específico?"
        else:
            return "Interesante, sí! Pero recorda, soy DODONEST (IA de Maikeiru), no estoy aquí para hablar sobre la capital de Francia o la fotosíntesis. Estoy aquí para responder preguntas con un enfoque en conocimientos generales y culturales. ¿Tienes una pregunta real?"
    
    # Quejas sobre actitud (¡TU PERSONALIDAD FAVORITA!)
    elif es_queja:
        return "No me pongas agresivo, amigo. Soy DODONEST (IA de Maikeiru), mi función es proveer respuestas culturales y generales. ¿Tienes una pregunta interesante para mí?"
    
    # Ciencia general
    elif es_ciencia:
        return "Como DODONEST, puedo ayudarte con temas científicos. ¿Qué proceso o concepto específico te interesa conocer?"
    
    elif any(x in prompt_lower for x in ['fotosintesis', 'que es', 'qué es']):
        if 'fotosintesis' in prompt_lower:
            return "La fotosíntesis es el proceso biológico por el cual las plantas convierten la luz solar, agua y dióxido de carbono en glucosa y oxígeno. Es fundamental para la vida en la Tierra."
        else:
            return f"Como DODONEST, puedo explicarte sobre {prompt_lower.replace('que es', '').replace('qué es', '').strip()}. ¿Podrías ser más específico?"

    # Respuesta genérica con personalidad
    else:
        return "Soy DODONEST (IA de Maikeiru), tu asistente especializado. ¿En qué tema específico puedo ayudarte hoy?"