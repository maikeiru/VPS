import re
from datetime import datetime

def formatear_fecha_amigable(timestamp_iso):
    """
    Convierte timestamp ISO a formato amigable.
    """
    try:
        dt = datetime.fromisoformat(timestamp_iso.replace('Z', '+00:00'))
        dias_diff = (datetime.now() - dt).days
        if dias_diff == 0:
            return "hoy"
        elif dias_diff == 1:
            return "ayer"
        elif dias_diff < 7:
            return f"hace {dias_diff} días"
        elif dias_diff < 30:
            semanas = dias_diff // 7
            return f"hace {semanas} semana{'s' if semanas > 1 else ''}"
        else:
            return dt.strftime("el %d/%m/%Y")
    except:
        return "hace algún tiempo"

def extraer_contexto_pregunta(pregunta):
    """
    Extrae el contexto de la pregunta original.
    """
    pregunta_lower = pregunta.lower()
    contextos = {
        'receta': ['receta', 'cocinar', 'ingredientes', 'comida', 'plato'],
        'idioma': ['ruso', 'inglés', 'español', 'traducir', 'idioma'],
        'poesía': ['poema', 'verso', 'literatura', 'poeta'],
        'general': ['fácil', 'rápido', 'simple', 'ayuda']
    }
    for tipo, palabras in contextos.items():
        if any(palabra in pregunta_lower for palabra in palabras):
            return f"cuando preguntaste por algo relacionado con {tipo}"
    return "en nuestra conversación anterior"

def dividir_texto_para_traduccion(texto, max_chars=400):
    """
    Divide texto largo en fragmentos para traducción por partes.
    """
    fragmentos = []
    lineas = texto.split('\n')
    fragmento_actual = ""
    for linea in lineas:
        if len(fragmento_actual + linea + '\n') > max_chars and fragmento_actual:
            fragmentos.append(fragmento_actual.strip())
            fragmento_actual = linea + '\n'
        else:
            fragmento_actual += linea + '\n'
    if fragmento_actual.strip():
        fragmentos.append(fragmento_actual.strip())
    return fragmentos

def verificar_respuesta_incompleta(respuesta, texto_original):
    """
    Verifica si una traducción está incompleta.
    """
    if not respuesta or len(respuesta.strip()) < 20:
        return True
    termina_mal = respuesta.rstrip().endswith(('...', '..', 'continúa', 'continua', 'sigue'))
    ratio_longitud = len(respuesta) / max(len(texto_original), 1)
    muy_corta = ratio_longitud < 0.3
    return termina_mal or muy_corta