import re

def recortar_respuesta_breve(respuesta, max_caracteres=160):
    """
    Recorta una respuesta para hacerla breve y directa.
    Elimina saludos, presentaciones y frases innecesarias.
    """
    # Limpia saltos y espacios raros
    respuesta = respuesta.replace("<br>", "\n").replace("\r", "").strip()
    
    # Aplicar múltiples pasadas de limpieza secuencial
    respuesta_limpia = respuesta
    
    # Primera pasada: eliminar frases completas específicas
    frases_completas = [
        r"^(?i)hola[,\s]*como\s+dodonest[,\s]*te\s+informo\s+que\s+",
        r"^(?i)como\s+dodonest[,\s]*te\s+informo\s+que\s+",
        r"^(?i)soy\s+dodonest[,\s]*te\s+informo\s+que\s+",
    ]
    
    for patron in frases_completas:
        nueva = re.sub(patron, "", respuesta_limpia, count=1).strip()
        if nueva != respuesta_limpia and len(nueva.strip()) > 10:
            respuesta_limpia = nueva
            break
    
    # Segunda pasada: eliminar frases individuales si no se eliminó nada antes
    if respuesta_limpia == respuesta:
        frases_individuales = [
            r"^(?i)hola[,\s]+",
            r"^(?i)como\s+dodonest[,\s]+",
            r"^(?i)soy\s+dodonest[,\s]+", 
            r"^(?i)te\s+informo\s+que\s+",
            r"^(?i)en\s+resumen[,\s]+",
            r"^(?i)en\s+conclusión[,\s]+",
            r"^(?i)para\s+finalizar[,\s]+",
        ]
        
        for patron in frases_individuales:
            nueva = re.sub(patron, "", respuesta_limpia, count=1).strip()
            if nueva != respuesta_limpia and len(nueva.strip()) > 10:
                respuesta_limpia = nueva
                break
    
    # Si quedó muy corto después de limpiar, usar el original
    if len(respuesta_limpia.strip()) < 10:
        respuesta_limpia = respuesta.strip()
    
    # Buscar la primera oración completa (hasta el punto)
    match = re.search(r"^([^.]*\.)", respuesta_limpia)
    if match:
        resultado = match.group(1).strip()
    else:
        # Si no hay punto, cortar por longitud de caracteres
        resultado = respuesta_limpia[:max_caracteres].strip()
    
    # Capitalizar primera letra si es necesario
    if resultado and not resultado[0].isupper():
        resultado = resultado[0].upper() + resultado[1:]
    
    return resultado