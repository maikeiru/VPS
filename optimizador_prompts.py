import re

class OptimizadorPrompts:
    """
    Optimiza prompts para reducir tokens manteniendo calidad:
    - Elimina redundancias
    - Comprime contexto largo
    - Mantiene información esencial
    - Optimiza para APIs específicas
    """
    
    def __init__(self):
        self.max_tokens_contexto = 800  # Límite para contexto
        self.max_tokens_historial = 400  # Límite para historial
        
    def optimizar_prompt_completo(self, mensaje, memoria, historial_chat, tipo_intencion="default"):
        """Optimiza el prompt completo según la intención detectada"""
        
        if tipo_intencion == "breve":
            return self._prompt_breve(mensaje)
        elif tipo_intencion == "ejemplo":
            return self._prompt_ejemplo(mensaje, memoria)
        elif tipo_intencion == "pasos":
            return self._prompt_pasos(mensaje)
        else:
            return self._prompt_optimizado_general(mensaje, memoria, historial_chat)
    
    def _prompt_breve(self, mensaje):
        """Prompt ultra-optimizado para respuestas breves"""
        return f"""Eres DODONEST (IA de Maikeiru). Respuesta en 1-2 frases máximo:

{mensaje}"""

    def _prompt_ejemplo(self, mensaje, memoria):
        """Prompt optimizado para ejemplos"""
        conocimientos_relevantes = self._extraer_conocimientos_relevantes(mensaje, memoria, limite=2)
        contexto_mini = "\n".join(conocimientos_relevantes) if conocimientos_relevantes else ""
        
        return f"""Eres DODONEST. Da un ejemplo concreto para: {mensaje}
{f"Contexto: {contexto_mini}" if contexto_mini else ""}"""

    def _prompt_pasos(self, mensaje):
        """Prompt optimizado para instrucciones paso a paso"""
        return f"""Eres DODONEST. Lista pasos numerados para: {mensaje}"""
    
    def _prompt_optimizado_general(self, mensaje, memoria, historial_chat):
        """Prompt general optimizado"""
        # Comprimir conocimientos
        conocimientos = self._comprimir_conocimientos(memoria, limite=5)
        
        # Comprimir historial
        historial = self._comprimir_historial(historial_chat, limite=2)
        
        # Construir prompt compacto
        prompt = "Eres DODONEST (IA de Maikeiru)."
        
        if historial:
            prompt += f"\nContexto: {historial}"
        
        if conocimientos:
            prompt += f"\nConocimientos: {conocimientos}"
        
        prompt += f"\n\nPregunta: {mensaje}\nRespuesta:"
        
        return prompt
    
    def _comprimir_conocimientos(self, memoria, limite=5):
        """Comprime conocimientos eliminando redundancias"""
        conocimientos_raw = memoria.get_conocimientos(limite)
        if not conocimientos_raw:
            return ""
        
        # Eliminar duplicados y redundancias
        conocimientos_unicos = []
        for c in conocimientos_raw:
            c_limpio = self._limpiar_texto(c)
            if c_limpio and not any(self._es_similar(c_limpio, existente) for existente in conocimientos_unicos):
                conocimientos_unicos.append(c_limpio)
        
        # Unir de forma compacta
        return " | ".join(conocimientos_unicos[:3])  # Solo top 3
    
    def _comprimir_historial(self, historial_chat, limite=2):
        """Comprime historial de chat"""
        if not historial_chat:
            return ""
        
        # Tomar solo las últimas entradas
        ultimas = historial_chat[-limite:] if len(historial_chat) > limite else historial_chat
        
        # Comprimir cada entrada
        entradas_comprimidas = []
        for entrada in ultimas:
            pregunta = self._truncar_texto(entrada['pregunta'], 50)
            respuesta = self._truncar_texto(entrada['respuesta'], 60)
            entradas_comprimidas.append(f"Q:{pregunta} A:{respuesta}")
        
        return " | ".join(entradas_comprimidas)
    
    def _extraer_conocimientos_relevantes(self, mensaje, memoria, limite=3):
        """Extrae solo conocimientos relevantes al mensaje"""
        conocimientos = memoria.get_conocimientos(10)
        relevantes = []
        
        palabras_mensaje = set(mensaje.lower().split())
        
        for c in conocimientos:
            palabras_conocimiento = set(c.lower().split())
            interseccion = len(palabras_mensaje.intersection(palabras_conocimiento))
            if interseccion > 0:
                relevantes.append((c, interseccion))
        
        # Ordenar por relevancia y tomar los mejores
        relevantes.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in relevantes[:limite]]
    
    def _limpiar_texto(self, texto):
        """Limpia y normaliza texto"""
        if not texto:
            return ""
        
        # Eliminar caracteres extra
        texto = re.sub(r'\s+', ' ', texto.strip())
        
        # Eliminar prefijos comunes redundantes
        prefijos_redundantes = [
            "Soy DODONEST, una IA creada por Maikeiru",
            "Como DODONEST",
            "DODONEST aquí",
        ]
        
        for prefijo in prefijos_redundantes:
            if texto.startswith(prefijo):
                texto = texto[len(prefijo):].strip()
        
        return texto
    
    def _truncar_texto(self, texto, max_chars):
        """Trunca texto manteniendo coherencia"""
        if len(texto) <= max_chars:
            return texto
        
        # Truncar por palabra completa
        truncado = texto[:max_chars].rsplit(' ', 1)[0]
        return truncado + "..."
    
    def _es_similar(self, texto1, texto2, threshold=0.7):
        """Verifica si dos textos son similares"""
        import difflib
        return difflib.SequenceMatcher(None, texto1, texto2).ratio() > threshold
    
    def calcular_tokens_ahorrados(self, prompt_original, prompt_optimizado):
        """Calcula tokens ahorrados por la optimización"""
        tokens_original = len(prompt_original.split()) * 1.3
        tokens_optimizado = len(prompt_optimizado.split()) * 1.3
        return max(0, int(tokens_original - tokens_optimizado))