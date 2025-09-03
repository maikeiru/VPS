import difflib
import json
import hashlib
from datetime import datetime, timedelta

class CacheInteligente:
    """Cache inteligente CORREGIDO - sin errores"""

    def __init__(self, maxsize=2000, similitud_threshold=0.95):
        self.cache = {}
        self.maxsize = maxsize
        self.similitud_threshold = similitud_threshold
        self.stats_ahorro = {
            'hits_exactos': 0,
            'hits_similares': 0,
            'misses': 0,
            'tokens_ahorrados_exactos': 0,
            'tokens_ahorrados_similares': 0,
            'consultas_totales': 0
        }
        self.archivo_stats = "cache_stats.json"
        self.cargar_stats()

    def _generar_key(self, pregunta, contexto=""):
        """Genera clave única para la pregunta + contexto"""
        contenido = f"{pregunta.lower().strip()}{contexto}"
        return hashlib.md5(contenido.encode()).hexdigest()

    def _calcular_tokens_estimados(self, texto):
        """Estima tokens de un texto (aprox 1.3 palabras = 1 token)"""
        if not isinstance(texto, str):
            return 0
        return int(len(texto.split()) * 1.3)

    def get(self, pregunta, contexto=""):
        """Obtiene respuesta del cache"""
        self.stats_ahorro['consultas_totales'] += 1

        # Buscar match exacto
        key_exacta = self._generar_key(pregunta, contexto)
        if key_exacta in self.cache:
            entry = self.cache[key_exacta]
            respuesta = entry['respuesta']

            if not isinstance(respuesta, str) or len(respuesta.strip()) < 5:
                del self.cache[key_exacta]
                return None, 'miss'

            tokens_ahorrados = self._calcular_tokens_estimados(respuesta)
            self.stats_ahorro['hits_exactos'] += 1
            self.stats_ahorro['tokens_ahorrados_exactos'] += tokens_ahorrados
            print(f"🎯 Cache HIT exacto: {tokens_ahorrados} tokens ahorrados")
            return respuesta, 'exacto'

        # No buscar similares por ahora - causa problemas
        self.stats_ahorro['misses'] += 1
        return None, 'miss'

    def set(self, pregunta, respuesta, contexto="", tokens_usados=0):
        """Guarda respuesta en cache"""
        if not isinstance(respuesta, str) or len(respuesta.strip()) < 10:
            return

        # Excluir respuestas genéricas
        respuesta_lower = respuesta.lower()
        frases_excluir = [
            "¿en qué puedo ayudarte?",
            "soy dodonest, tu asistente",
            "hola! soy dodonest"
        ]
        if any(frase in respuesta_lower for frase in frases_excluir):
            return

        # Limpiar cache si está lleno
        if len(self.cache) >= self.maxsize:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        # Guardar en cache
        key = self._generar_key(pregunta, contexto)
        self.cache[key] = {
            'pregunta_original': pregunta,
            'respuesta': respuesta,
            'contexto': contexto,
            'timestamp': datetime.now().isoformat(),
            'tokens_usados': tokens_usados
        }

    def get_estadisticas_ahorro(self):
        """Estadísticas"""
        total_tokens_ahorrados = (
            self.stats_ahorro['tokens_ahorrados_exactos'] +
            self.stats_ahorro['tokens_ahorrados_similares']
        )
        total_hits = self.stats_ahorro['hits_exactos'] + self.stats_ahorro['hits_similares']
        total_consultas = self.stats_ahorro['consultas_totales']

        hit_rate = (total_hits / total_consultas * 100) if total_consultas > 0 else 0

        return {
            'total_tokens_ahorrados': total_tokens_ahorrados,
            'hit_rate_porcentaje': hit_rate,
            'hits_exactos': self.stats_ahorro['hits_exactos'],
            'hits_similares': self.stats_ahorro['hits_similares'],
            'total_consultas': total_consultas,
            'ahorro_estimado_usd': total_tokens_ahorrados * 0.00002
        }

    def cargar_stats(self):
        """Carga estadísticas persistentes"""
        try:
            with open(self.archivo_stats, 'r') as f:
                self.stats_ahorro.update(json.load(f))
        except FileNotFoundError:
            pass

    def guardar_stats(self):
        """Guarda estadísticas en disco"""
        with open(self.archivo_stats, 'w') as f:
            json.dump(self.stats_ahorro, f, indent=2)

    def limpiar_cache_antiguo(self, dias=7):
        """Limpia cache antiguo"""
        print("🧹 Cache limpiado manualmente")
        self.cache.clear()
