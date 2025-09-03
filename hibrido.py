import difflib
from datetime import datetime

class SistemaHibridoMejorado:
    """
    Sistema híbrido optimizado que maximiza el ahorro de tokens:
    - Evaluación inteligente de Llama3
    - Decisiones basadas en métricas reales
    - Caché inteligente integrado
    - Optimización de prompts automática
    """

    def __init__(self, sistema_base, cache_inteligente, optimizador_prompts):
        self.sistema = sistema_base
        self.cache = cache_inteligente
        self.optimizador = optimizador_prompts
        self.stats_detalladas = {
            'respuestas_llama3_exitosas': 0,
            'respuestas_api_completas': 0,
            'respuestas_cache': 0,
            'tokens_ahorrados_llama3': 0,
            'tokens_ahorrados_cache': 0,
            'tokens_ahorrados_optimizacion': 0,
            'calidad_promedio_llama3': 0.0,
            'decisiones_inteligentes': 0
        }

    def generar_respuesta_optimizada(self, mensaje, motor="hibrido"):
        """Genera respuesta optimizada con máximo ahorro de tokens"""
        
        # 1. PRIMERA VERIFICACIÓN: Cache inteligente
        respuesta_cache, tipo_cache = self.cache.get(mensaje)
        if respuesta_cache:
            self._registrar_uso_cache(tipo_cache, respuesta_cache)
            return respuesta_cache

        # 2. ANALIZAR INTENCIÓN para optimizar estrategia
        from intencion import analizar_intencion
        tipo_intencion = analizar_intencion(mensaje)
        
        # 3. EVALUAR LLAMA3 LOCAL primero (siempre es gratis)
        contenido_llama3 = self._generar_llama3_optimizado(mensaje, tipo_intencion)
        score_llama3, detalles_evaluacion = self._evaluar_llama3_mejorado(contenido_llama3, mensaje)
        
        print(f"📊 Evaluación Llama3: {score_llama3:.2f} - {detalles_evaluacion}")

        # 4. DECISIÓN INTELIGENTE basada en score y tipo
        if self._debe_usar_llama3(score_llama3, tipo_intencion, mensaje):
            respuesta_final = self._procesar_respuesta_llama3(contenido_llama3, mensaje, score_llama3)
            metodo = "llama3_directo"
            tokens_ahorrados = self._estimar_tokens_ahorrados_llama3(mensaje)
            self.stats_detalladas['tokens_ahorrados_llama3'] += tokens_ahorrados
            self.stats_detalladas['respuestas_llama3_exitosas'] += 1
            
        elif score_llama3 > 0.3:  # Usar como base para API
            respuesta_final, tokens_usados = self._mejorar_con_api(contenido_llama3, mensaje, score_llama3)
            metodo = "hibrido_mejorado"
            tokens_ahorrados = self._calcular_ahorro_hibrido(contenido_llama3, tokens_usados)
            self.stats_detalladas['tokens_ahorrados_llama3'] += tokens_ahorrados
            
        else:  # API completa pero optimizada
            respuesta_final, tokens_usados = self._generar_api_optimizada(mensaje, tipo_intencion)
            metodo = "api_optimizada"
            self.stats_detalladas['respuestas_api_completas'] += 1

        # 5. GUARDAR EN CACHE para futuras consultas
        self.cache.set(mensaje, respuesta_final)
        
        # 6. REGISTRAR ESTADÍSTICAS
        self._registrar_estadisticas(mensaje, respuesta_final, metodo, score_llama3)
        
        print(f"✅ Respuesta generada: {metodo} | Score Llama3: {score_llama3:.2f}")
        return respuesta_final

    def _generar_llama3_optimizado(self, mensaje, tipo_intencion):
        """Genera respuesta con Llama3 usando prompt optimizado"""
        prompt_optimizado = self.optimizador.optimizar_prompt_completo(
            mensaje, self.sistema.memoria, self.sistema.historial_chat, tipo_intencion
        )
        
        # Calcular ahorro por optimización
        prompt_original = self.sistema.construir_prompt(mensaje, self.sistema.memoria, self.sistema.historial_chat)
        tokens_ahorrados = self.optimizador.calcular_tokens_ahorrados(prompt_original, prompt_optimizado)
        self.stats_detalladas['tokens_ahorrados_optimizacion'] += tokens_ahorrados
        
        from llama3_api import llama3_query
        return llama3_query(prompt_optimizado)

    def _evaluar_llama3_mejorado(self, contenido, mensaje):
        """Evaluación mejorada universal para cualquier tema"""
        if not contenido or len(contenido.strip()) < 15:
            return 0.0, "Contenido muy corto o vacío"

        # Detectar categoría del tema
        mensaje_lower = mensaje.lower()
        categoria = "general"
        if any(x in mensaje_lower for x in ['fotosintesis', 'quimica', 'fisica', 'biologia', 'matematicas']):
            categoria = "ciencia"
        elif any(x in mensaje_lower for x in ['capital', 'pais', 'ciudad', 'geografia']):
            categoria = "geografia"
        elif any(x in mensaje_lower for x in ['historia', 'guerra', 'siglo', 'epoca']):
            categoria = "historia"
        elif any(x in mensaje_lower for x in ['programacion', 'codigo', 'software', 'python']):
            categoria = "tecnologia"

        puntuaciones = {}
        palabras = len(contenido.split())
        
        # 1. Longitud adaptativa según categoría
        if categoria == "ciencia":
            if palabras < 10: puntuaciones['longitud'] = 0.3
            elif palabras < 30: puntuaciones['longitud'] = 0.8
            elif palabras < 80: puntuaciones['longitud'] = 1.0
            else: puntuaciones['longitud'] = 0.9
        elif categoria == "geografia":
            if palabras < 5: puntuaciones['longitud'] = 0.2
            elif palabras < 15: puntuaciones['longitud'] = 1.0
            else: puntuaciones['longitud'] = 0.8
        else:  # general
            if palabras < 8: puntuaciones['longitud'] = 0.4
            elif palabras < 25: puntuaciones['longitud'] = 0.9
            elif palabras < 60: puntuaciones['longitud'] = 1.0
            else: puntuaciones['longitud'] = 0.8

        # 2. Detección de basura mejorada
        puntuaciones['sin_basura'] = 0.0 if self._contiene_basura_mejorado(contenido) else 1.0

        # 3. Relevancia semántica mejorada
        puntuaciones['relevancia'] = self._calcular_relevancia_por_categoria(contenido, mensaje, categoria)

        # 4. Coherencia estructural
        puntuaciones['coherencia'] = self._evaluar_coherencia_mejorada(contenido)

        # 5. Personalidad DODONEST (NUEVO - MUY IMPORTANTE)
        puntuaciones['personalidad'] = self._evaluar_personalidad_dodonest(contenido)

        # 6. Completitud según categoría
        puntuaciones['completitud'] = self._evaluar_completitud_por_categoria(contenido, mensaje, categoria)

        # Cálculo final con pesos adaptativos
        if categoria == "ciencia":
            score_final = (puntuaciones['longitud'] * 0.15 + puntuaciones['sin_basura'] * 0.2 + 
                          puntuaciones['relevancia'] * 0.25 + puntuaciones['coherencia'] * 0.15 + 
                          puntuaciones['personalidad'] * 0.15 + puntuaciones['completitud'] * 0.1)
        elif categoria == "geografia":
            score_final = (puntuaciones['longitud'] * 0.1 + puntuaciones['sin_basura'] * 0.3 + 
                          puntuaciones['relevancia'] * 0.3 + puntuaciones['coherencia'] * 0.1 + 
                          puntuaciones['personalidad'] * 0.15 + puntuaciones['completitud'] * 0.05)
        else:  # general
            score_final = (puntuaciones['longitud'] * 0.15 + puntuaciones['sin_basura'] * 0.2 + 
                          puntuaciones['relevancia'] * 0.2 + puntuaciones['coherencia'] * 0.15 + 
                          puntuaciones['personalidad'] * 0.25 + puntuaciones['completitud'] * 0.05)

        # Bonus por respuestas excepcionales
        if score_final > 0.85 and palabras > 20 and puntuaciones['personalidad'] > 0.7:
            score_final = min(1.0, score_final + 0.1)

        detalles = f"Cat:{categoria} Long:{puntuaciones['longitud']:.1f} Pers:{puntuaciones['personalidad']:.1f} Rel:{puntuaciones['relevancia']:.1f}"
        
        return score_final, detalles

    def _calcular_relevancia_por_categoria(self, contenido, mensaje, categoria):
        """Relevancia mejorada por categoría"""
        palabras_mensaje = set(mensaje.lower().split())
        palabras_contenido = set(contenido.lower().split())
        
        if not palabras_mensaje:
            return 0.0
        
        # Coincidencias básicas
        coincidencias = len(palabras_mensaje.intersection(palabras_contenido))
        relevancia_basica = coincidencias / len(palabras_mensaje)
        
        # Bonus por palabras clave específicas de categoría
        if categoria == "ciencia":
            palabras_clave = ['proceso', 'energia', 'molecular', 'quimico', 'celular']
            bonus = sum(1 for p in palabras_clave if p in contenido.lower()) * 0.1
        elif categoria == "geografia":
            palabras_clave = ['ubicada', 'situada', 'region', 'territorio']
            bonus = sum(1 for p in palabras_clave if p in contenido.lower()) * 0.15
        else:
            bonus = 0
        
        return min(1.0, relevancia_basica * 1.2 + bonus)

    def _evaluar_personalidad_dodonest(self, contenido):
        """Evalúa si mantiene la personalidad característica de DODONEST"""
        contenido_lower = contenido.lower()
        puntuacion = 0.5  # Base neutral
        
        # Identidad correcta (CRÍTICO)
        if 'dodonest' in contenido_lower:
            puntuacion += 0.3
        if 'maikeiru' in contenido_lower:
            puntuacion += 0.2
        
        # Personalidad característica (el tono que te gusta!)
        frases_personalidad = [
            'amigo', 'vale', 'interesante', 'real', 'pregunta', 
            'función', 'proveer', 'brindo', 'aquí para'
        ]
        personalidad_detectada = sum(1 for frase in frases_personalidad if frase in contenido_lower)
        puntuacion += min(0.3, personalidad_detectada * 0.1)
        
        # Penalizar identidades incorrectas
        identidades_malas = ['soy bard', 'soy gpt', 'google', 'openai']
        if any(bad in contenido_lower for bad in identidades_malas):
            puntuacion = 0.0
        
        return min(1.0, puntuacion)

    def _evaluar_completitud_por_categoria(self, contenido, mensaje, categoria):
        """Evalúa completitud según el tipo de pregunta"""
        if "?" in mensaje:  # Es pregunta
            if categoria == "geografia" and len(contenido.split()) >= 3:
                return 1.0
            elif categoria == "ciencia" and len(contenido.split()) >= 15:
                return 1.0
            elif len(contenido.split()) >= 8:
                return 0.8
            return 0.4
        return 0.8  # Afirmaciones son más difíciles de evaluar

    def _contiene_basura_mejorado(self, contenido):
        """Detección mejorada de contenido basura"""
        basura_patterns = [
            "géro", "fresí", "DAKO", "[RNG]", "want to learn",
            "abajo con hace", "nuevo donde que", "llegar una entre",
            "sorry, i can't", "no puedo ayudar", "como modelo de",
            "soy un modelo", "como ia de google", "soy bard"
        ]
        contenido_lower = contenido.lower()
        
        # Verificar patrones de basura
        for pattern in basura_patterns:
            if pattern.lower() in contenido_lower:
                return True
        
        # Verificar repeticiones extrañas
        palabras = contenido_lower.split()
        if len(palabras) > 5:
            for i in range(len(palabras) - 2):
                if palabras[i] == palabras[i+1] == palabras[i+2]:
                    return True
        
        return False

    def _calcular_relevancia_mejorada(self, contenido, mensaje):
        """Cálculo mejorado de relevancia"""
        palabras_mensaje = set(mensaje.lower().split())
        palabras_contenido = set(contenido.lower().split())
        
        if not palabras_mensaje:
            return 0.0
        
        # Coincidencias exactas
        coincidencias_exactas = len(palabras_mensaje.intersection(palabras_contenido))
        relevancia_basica = coincidencias_exactas / len(palabras_mensaje)
        
        # Bonus por coincidencias de palabras clave importantes
        palabras_importantes = [p for p in palabras_mensaje if len(p) > 4]
        coincidencias_importantes = len(set(palabras_importantes).intersection(palabras_contenido))
        bonus_importantes = coincidencias_importantes * 0.2 if palabras_importantes else 0
        
        return min(1.0, relevancia_basica * 1.5 + bonus_importantes)

    def _evaluar_coherencia_mejorada(self, contenido):
        """Evaluación mejorada de coherencia"""
        oraciones = [o.strip() for o in contenido.split('.') if len(o.strip()) > 5]
        
        if len(oraciones) == 0:
            return 0.0
        elif len(oraciones) == 1:
            return 0.8 if len(oraciones[0]) > 20 else 0.5
        
        # Verificar transiciones lógicas
        coherencia = 0.9
        for i in range(1, len(oraciones)):
            if not self._hay_conexion_logica(oraciones[i-1], oraciones[i]):
                coherencia -= 0.2
        
        return max(0.0, coherencia)

    def _hay_conexion_logica(self, oracion1, oracion2):
        """Verifica si hay conexión lógica entre oraciones"""
        conectores = ['por lo tanto', 'además', 'sin embargo', 'también', 'esto', 'eso', 'este']
        return any(conector in oracion2.lower() for conector in conectores)

    def _evaluar_completitud(self, contenido, mensaje):
        """Evalúa si la respuesta está completa"""
        if "¿" in mensaje:  # Es una pregunta
            if "?" in contenido or "." in contenido:
                return 1.0
            return 0.5
        return 0.8  # Para afirmaciones es más difícil evaluar completitud

    def _verificar_identidad_dodonest(self, contenido):
        """Verifica que mantenga la identidad DODONEST correcta"""
        contenido_lower = contenido.lower()
        
        # Penalizar identidades incorrectas
        identidades_malas = ['soy bard', 'soy gpt', 'soy claude', 'como modelo de google']
        if any(bad in contenido_lower for bad in identidades_malas):
            return 0.0
        
        # Premiar identidad correcta
        identidades_buenas = ['dodonest', 'maikeiru']
        if any(good in contenido_lower for good in identidades_buenas):
            return 1.0
        
        return 0.7  # Neutral

    def _debe_usar_llama3(self, score, tipo_intencion, mensaje):
        """Decisión CORREGIDA sobre usar Llama3 vs APIs"""
        
        # Detectar categoría del mensaje
        categoria = self._detectar_categoria_mensaje(mensaje)
        
        # Umbrales MÁS ESTRICTOS
        umbrales = {
            'identidad': 0.75,    # Preguntas sobre identidad - más permisivo
            'agradecimiento': 0.70,  # Agradecimientos simples
            'saludo': 0.80,       # Solo saludos simples
            'geografia': 0.95,    # Hechos geográficos simples
            'historia': 0.98,     # Historia casi NUNCA Llama3
            'detallado': 0.99,    # Resúmenes detallados NUNCA Llama3
            'ciencia': 0.95,      # Ciencia requiere precisión
            'default': 0.85       # Por defecto más estricto
        }
        
        umbral = umbrales.get(categoria, umbrales['default'])
        
        print(f"🎯 Categoría: {categoria} | Umbral: {umbral} | Score: {score:.2f}")
        
        # Casos donde FORZAR API (ignorar score)
        if categoria in ['historia', 'detallado'] or \
           any(x in mensaje.lower() for x in ['revolucion', 'resumen detallado', 'explicame', 'cuentame']):
            print(f"🔄 Forzando API para {categoria} - contenido complejo")
            return False  # Usar API
        if categoria in ['ciencia', 'geografia', 'tecnologia'] and score < 0.8:
            print(f"🔄 Forzando API para {categoria} por precisión requerida")
            return False  # Usar API
            
        # Saludos muy simples pueden usar Llama3 con menor exigencia
        if categoria == 'saludo' and len(mensaje.split()) <= 3:
            return True
            
        # Decisión normal por score
        return score >= umbral
            
            

    def _detectar_categoria_mensaje(self, mensaje):
        """Detecta categoría REAL del mensaje - ULTRA CORREGIDO v3"""
        mensaje_lower = mensaje.lower().strip()
        
        # Identidad y preguntas personales (PRIMERA PRIORIDAD)
        if any(x in mensaje_lower for x in ['quien soy', 'quién soy', 'sabes quien', 'soy maikeiru']):
            return "identidad"
        elif any(x in mensaje_lower for x in ['tu creador', 'tu amo', 'soy tu']):
            return "identidad"
        elif any(x in mensaje_lower for x in ['perfecto', 'gracias', 'muchas gracias']):
            return "agradecimiento"
        
        # Saludos ESPECÍFICOS (segunda prioridad)
        elif mensaje_lower in ['hola', 'hey', 'saludos', 'buenas', 'hola!', 'hello']:
            return "saludo"
        
        # Como estás, qué tal (tercera prioridad)  
        elif any(x in mensaje_lower for x in ['como estas', 'cómo estás', 'que tal', 'qué tal']):
            return "saludo"
        
        # Historia ESPECÍFICA (ANTES de ciencia)
        elif any(x in mensaje_lower for x in ['revolucion', 'imperio romano', 'guerra mundial', 'historia']):
            return "historia"
        elif any(x in mensaje_lower for x in ['napoleon', 'siglo', 'epoca', 'reino']):
            return "historia"
        
        # Ciencia ESPECÍFICA
        elif any(x in mensaje_lower for x in ['fotosintesis', 'fotosíntesis', 'quimica', 'fisica', 'biologia']):
            return "ciencia"
        elif any(x in mensaje_lower for x in ['proceso', 'molecular', 'celular']):
            return "ciencia"
        
        # Geografía ESPECÍFICA
        elif any(x in mensaje_lower for x in ['capital de', 'capital del', 'ciudad de']):
            return "geografia"
        elif any(x in mensaje_lower for x in ['francia', 'españa', 'mexico', 'pais', 'país']):
            return "geografia"
        
        # Resúmenes detallados
        elif any(x in mensaje_lower for x in ['detalladamente', 'explicame detalladamente', 'resumen detallado']):
            return "detallado"
        elif any(x in mensaje_lower for x in ['4 parrafos', 'varios parrafos', 'mas informacion']):
            return "detallado"
        
        else:
            return "general"

    def _procesar_respuesta_llama3(self, contenido, mensaje, score):
        """Procesa y mejora la respuesta de Llama3 si es necesario"""
        # Si el score es excelente, usar tal como está
        if score > 0.9:
            return contenido
        
        # Aplicar mejoras menores
        respuesta_mejorada = self._aplicar_mejoras_menores(contenido)
        return respuesta_mejorada

    def _aplicar_mejoras_menores(self, contenido):
        """Aplica mejoras menores sin usar API"""
        # Limpiar formato
        contenido = contenido.strip()
        
        # Asegurar mayúscula inicial
        if contenido and not contenido[0].isupper():
            contenido = contenido[0].upper() + contenido[1:]
        
        # Asegurar punto final si no lo tiene
        if contenido and not contenido.endswith(('.', '!', '?')):
            contenido += '.'
        
        return contenido

    def _mejorar_con_api(self, contenido_base, mensaje, score):
        """API mejorada - DEVOLVER TUPLA CORRECTA"""
        prompt = f"Eres DODONEST (IA de Maikeiru). Responde: {mensaje}"
        
        try:
            resultado = self.sistema.llamar_openai_seguro(prompt)
            if isinstance(resultado, tuple) and resultado[0] and isinstance(resultado[0], str):
                return resultado[0], resultado[1]  # ✅ DEVOLVER TUPLA
        except Exception as e:
            print(f"❌ OpenAI real falló: {e}")
        
        print("🔄 Usando Llama3 como respaldo")
        from llama3_api import llama3_query
        respuesta = llama3_query(prompt)
        return respuesta, 0  # ✅ DEVOLVER TUPLA TAMBIÉN

    def _generar_api_optimizada(self, mensaje, tipo_intencion):
        """Genera respuesta usando API pero con prompt optimizado"""
        prompt_optimizado = self.optimizador.optimizar_prompt_completo(
            mensaje, self.sistema.memoria, self.sistema.historial_chat, tipo_intencion
        )
        
        return self.sistema.llamar_openai_seguro(prompt_optimizado)
    
    def _generar_api_inteligente(self, mensaje, tipo_intencion):
        """Genera respuesta con API pero de forma más inteligente"""
        
        # Decidir qué API usar según el caso
        if "tiempo real" in mensaje.lower() or "noticias" in mensaje.lower():
            # Para info en tiempo real, usar OpenAI
            prompt = f"Como DODONEST (IA de Maikeiru), responde: {mensaje}"
            return self.sistema.llamar_openai_seguro(prompt)
        
        elif any(x in mensaje.lower() for x in ['ciencia', 'matematicas', 'tecnico']):
            # Para temas técnicos, usar Gemini (mejor en ciencias)
            prompt = f"Eres DODONEST (IA de Maikeiru). Responde técnicamente: {mensaje}"
            return self.sistema.llamar_gemini_seguro(prompt)
        
        else:
            # Para conversación general, alternar APIs
            import random
            if random.choice([True, False]):
                prompt = f"Como DODONEST (IA de Maikeiru), responde naturalmente: {mensaje}"
                return self.sistema.llamar_openai_seguro(prompt)
            else:
                prompt = f"Eres DODONEST (IA de Maikeiru). Responde: {mensaje}"
                return self.sistema.llamar_gemini_seguro(prompt)
    
    def _estimar_tokens_ahorrados_llama3(self, mensaje):
        """Estima tokens ahorrados al usar Llama3 en lugar de API"""
        # Estimación conservadora: prompt + respuesta típica
        tokens_estimados_api = len(mensaje.split()) * 1.3 + 200  # prompt + respuesta media
        return int(tokens_estimados_api)

    def _calcular_ahorro_hibrido(self, contenido_base, tokens_usados_api):
        """Calcula ahorro en modo híbrido"""
        tokens_base = len(contenido_base.split()) * 1.3
        # El ahorro es la diferencia entre generar todo desde cero vs mejorar la base
        tokens_generacion_completa_estimada = tokens_usados_api * 1.5
        return max(0, int(tokens_generacion_completa_estimada - tokens_usados_api))

    def _registrar_uso_cache(self, tipo_cache, respuesta):
        """Registra uso del cache"""
        tokens_ahorrados = len(respuesta.split()) * 1.3
        self.stats_detalladas['respuestas_cache'] += 1
        self.stats_detalladas['tokens_ahorrados_cache'] += tokens_ahorrados
        self.cache.guardar_stats()

    def _encontrar_similar(self, pregunta_nueva):
        """Busca pregunta similar en cache usando difflib"""
        mejor_match = None
        mejor_ratio = 0
        
        for key, data in self.cache.items():
            pregunta_cache = data['pregunta_original']
            ratio = difflib.SequenceMatcher(None, 
                                          pregunta_nueva.lower(), 
                                          pregunta_cache.lower()).ratio()
            
            # AUMENTAR threshold de 0.8 a 0.95 para más precisión
            if ratio > mejor_ratio and ratio >= 0.95:  # CAMBIO AQUÍ
                mejor_ratio = ratio
                mejor_match = (key, data, ratio)
        
        return mejor_match
    
    def _registrar_estadisticas(self, mensaje, respuesta, metodo, score_llama3):
        """Registra estadísticas detalladas"""
        self.stats_detalladas['decisiones_inteligentes'] += 1
        
        # Actualizar calidad promedio de Llama3
        if metodo.startswith('llama3'):
            total_llama3 = self.stats_detalladas['respuestas_llama3_exitosas']
            self.stats_detalladas['calidad_promedio_llama3'] = (
                (self.stats_detalladas['calidad_promedio_llama3'] * (total_llama3 - 1) + score_llama3) / total_llama3
            )

    def get_dashboard_ahorro(self):
        """Retorna dashboard completo de ahorro de tokens"""
        stats_cache = self.cache.get_estadisticas_ahorro()
        
        total_tokens_ahorrados = (
            self.stats_detalladas['tokens_ahorrados_llama3'] +
            self.stats_detalladas['tokens_ahorrados_cache'] +
            self.stats_detalladas['tokens_ahorrados_optimizacion']
        )
        
        ahorro_estimado_usd = total_tokens_ahorrados * 0.00002
        
        return f"""
🎯 DASHBOARD AHORRO DE TOKENS - DODONEST
{'='*50}
💰 AHORRO TOTAL: {total_tokens_ahorrados:,.0f} tokens (${ahorro_estimado_usd:.4f} USD)

📊 DISTRIBUCIÓN DEL AHORRO:
   🤖 Llama3 Local: {self.stats_detalladas['tokens_ahorrados_llama3']:,.0f} tokens
   🎯 Cache Inteligente: {self.stats_detalladas['tokens_ahorrados_cache']:,.0f} tokens  
   ⚡ Optimización Prompts: {self.stats_detalladas['tokens_ahorrados_optimizacion']:,.0f} tokens

🔄 MÉTODOS DE RESPUESTA:
   ✅ Llama3 Exitoso: {self.stats_detalladas['respuestas_llama3_exitosas']} respuestas
   🎯 Cache Hits: {self.stats_detalladas['respuestas_cache']} respuestas
   🔗 APIs Necesarias: {self.stats_detalladas['respuestas_api_completas']} respuestas

📈 MÉTRICAS DE CALIDAD:
   🎯 Cache Hit Rate: {stats_cache['hit_rate_porcentaje']:.1f}%
   🤖 Calidad Promedio Llama3: {self.stats_detalladas['calidad_promedio_llama3']:.2f}/1.0
   🧠 Decisiones Inteligentes: {self.stats_detalladas['decisiones_inteligentes']}

💡 RECOMENDACIONES:
   {"✅ Excelente ahorro de tokens!" if total_tokens_ahorrados > 1000 else "🔄 Continúa usando el sistema para más ahorro"}
   {"✅ Cache funcionando perfectamente!" if stats_cache['hit_rate_porcentaje'] > 20 else "💡 El cache mejorará con más uso"}
   {"✅ Llama3 está generando respuestas de calidad!" if self.stats_detalladas['calidad_promedio_llama3'] > 0.7 else "⚠️ Ajusta umbrales de Llama3 para mejor calidad"}
"""

    # MÉTODO PARA COMPATIBILIDAD CON TU MAIN.PY ACTUAL
    def generar_respuesta_hibrida(self, mensaje, motor="hibrido"):
        """Alias para compatibilidad con tu código actual"""
        return self.generar_respuesta_optimizada(mensaje, motor)