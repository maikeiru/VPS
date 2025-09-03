import re

def analizar_intencion(mensaje):
    mensaje = mensaje.lower()
    
    # Detectar peticiones de resumen (AMPLIADO)
    if any(x in mensaje for x in [
        "resumen", "resumir", "resume", "en pocas palabras", "brevemente",
        "qué es", "define", "definición", "explicar brevemente", "explica breve",
        "dime qué es", "que es", "explícame qué", "cuéntame qué", "breve",
        "resumido", "¿qué es", "qué significa", "explica corto", "más corto", 
        "más concreto", "conciso"
    ]):
        return "breve"
        
    if any(x in mensaje for x in [
        "resumen extenso", "detallado", "explica a fondo", "análisis",
        "completo", "profundo", "en detalle", "extensamente"
    ]):
        return "extenso"
        
    if any(x in mensaje for x in [
        "ejemplo", "dame ejemplos", "pon ejemplos", "caso práctico", "por ejemplo"
    ]):
        return "ejemplo"
        
    if any(x in mensaje for x in [
        "paso a paso", "cómo se hace", "guía", "instrucciones", "tutorial"
    ]):
        return "pasos"
        
    if any(x in mensaje for x in [
        "comparación", "diferencia", "similaridad", "compara", "versus", "vs"
    ]):
        return "comparacion"
        
    # Detecta insatisfacción o petición de aclaración
    if any(x in mensaje for x in [
        "no entendí", "eso no", "demasiado largo", "más simple",
        "más claro", "más explicado", "explícame mejor"
    ]):
        return "insatisfaccion"
    
    if any(x in mensaje for x in [
        "buscar en internet", "busca en la web", "noticias actuales", "últimas noticias",
        "web", "google", "bing", "investiga en internet"
    ]):
        return "websearch"
        
    return "default"