"""
Módulo de integración de modelos multimodales optimizado para Windows y CUDA

Este módulo proporciona una implementación optimizada que funciona correctamente
en Windows con CUDA, evitando problemas de compatibilidad con cuantización.
"""

import os
import json
import logging
import traceback  # Importación correcta de traceback
import torch
from PIL import Image
from pathlib import Path

# Importar utilidades de plataforma
from src.platform_utils import normalize_path, ensure_dir, get_platform

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_multimodal.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("llm_multimodal")

class MultimodalLLMEngine:
    """
    Motor de integración de modelos de lenguaje multimodales optimizado para Windows y CUDA.
    """
    
    def __init__(self, model_name="microsoft/git-base", load_in_8bit=True, load_in_4bit=False, cache_dir=None):
        """
        Inicializa el motor multimodal.
        
        Args:
            model_name: Nombre del modelo a utilizar
            load_in_8bit: Si se debe cargar el modelo en precisión de 8 bits
            load_in_4bit: Si se debe cargar el modelo en precisión de 4 bits
            cache_dir: Directorio de caché para modelos
        """
        self.model_name = model_name
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.cache_dir = normalize_path(cache_dir) if cache_dir else None
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Verificar si estamos en Windows
        self.is_windows = get_platform() == 'windows'
        
        logger.info(f"MultimodalLLMEngine inicializado. Modelo: {model_name}")
        logger.info(f"Dispositivo: {self.device}")
        logger.info(f"Cuantización: {'8-bit' if load_in_8bit else '4-bit' if load_in_4bit else 'ninguna'}")
        
        # Cargar modelo
        self._load_model()
    
    def _load_model(self):
        """
        Carga el modelo multimodal optimizado para Windows y CUDA.
        """
        try:
            logger.info(f"Cargando modelo: {self.model_name}")
            
            # Importar bibliotecas necesarias
            import torch
            
            # Limpiar caché de CUDA si está disponible
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Configurar opciones de carga
            load_options = {
                "torch_dtype": torch.float32,  # Usar float32 para mayor compatibilidad
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "ignore_mismatched_sizes": True  # Ignorar diferencias de tamaño para evitar errores
            }
            
            if self.cache_dir:
                ensure_dir(self.cache_dir)
                load_options["cache_dir"] = self.cache_dir
            
            # IMPORTANTE: Desactivar cuantización solo en Windows
            if self.is_windows:
                # En Windows, desactivar cuantización para evitar problemas con bitsandbytes
                self.load_in_8bit = False
                self.load_in_4bit = False
                logger.info("Cuantización desactivada en Windows para evitar problemas de compatibilidad")
            else:
                # En otros sistemas, mantener la configuración original
                if self.load_in_8bit and self.device == "cuda":
                    load_options["load_in_8bit"] = True
                elif self.load_in_4bit and self.device == "cuda":
                    load_options["load_in_4bit"] = True
            
            # En Windows, optimizar uso de memoria
            if self.is_windows:
                os.environ["OMP_NUM_THREADS"] = "1"
                os.environ["MKL_NUM_THREADS"] = "1"
                torch.set_num_threads(1)
            
            # Cargar modelo según su tipo
            if "git" in self.model_name.lower():
                # Cargar GIT (más ligero y compatible)
                from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
                
                self.processor = AutoProcessor.from_pretrained(self.model_name, **load_options)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_options)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **load_options)
                logger.info("Modelo GIT cargado correctamente")
            elif "blip" in self.model_name.lower():
                # Cargar BLIP2 (con manejo especial para Windows)
                from transformers import AutoProcessor, BlipForConditionalGeneration, AutoTokenizer
                
                self.processor = AutoProcessor.from_pretrained(self.model_name, **load_options)
                self.model = BlipForConditionalGeneration.from_pretrained(self.model_name, **load_options)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **load_options)
                logger.info("Modelo BLIP2 cargado correctamente")
            else:
                # Intentar cargar otros modelos multimodales
                try:
                    from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
                    
                    self.processor = AutoProcessor.from_pretrained(self.model_name, **load_options)
                    self.model = AutoModelForVision2Seq.from_pretrained(self.model_name, **load_options)
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **load_options)
                    logger.info("Modelo cargado correctamente usando AutoModelForVision2Seq")
                except Exception as e:
                    logger.error(f"Error al cargar modelo específico: {str(e)}")
                    
                    # Intentar con modelo alternativo
                    alt_model = "microsoft/git-base"
                    logger.info(f"Intentando con modelo alternativo: {alt_model}")
                    
                    from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
                    
                    self.processor = AutoProcessor.from_pretrained(alt_model, **load_options)
                    self.model = AutoModelForCausalLM.from_pretrained(alt_model, **load_options)
                    self.tokenizer = AutoTokenizer.from_pretrained(alt_model, **load_options)
                    logger.info("Modelo alternativo cargado correctamente")
            
            logger.info("Modelo cargado correctamente")
            
        except Exception as e:
            logger.error(f"Error al cargar modelo: {str(e)}")
            logger.error(traceback.format_exc())
            self.model = None
            self.processor = None
            self.tokenizer = None
    
    def unload_model(self):
        """
        Descarga el modelo para liberar memoria.
        """
        try:
            if self.model:
                del self.model
                self.model = None
            
            if self.processor:
                del self.processor
                self.processor = None
            
            # Limpiar caché de CUDA si está disponible
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Modelo descargado correctamente")
            
        except Exception as e:
            logger.error(f"Error al descargar modelo: {str(e)}")
    
    def analyze_step(self, step, context=""):
        """
        Analiza un paso de la sesión.
        
        Args:
            step: Datos del paso
            context: Contexto de la sesión
            
        Returns:
            dict: Resultados del análisis
        """
        try:
            if not self.model or not self.processor:
                logger.error("Modelo no disponible")
                return {
                    "success": False,
                    "description": "No se pudo analizar el paso: modelo no disponible"
                }
            
            # Obtener ruta de la imagen
            image_path = step.get("image", {}).get("path", "")
            
            if not image_path or not os.path.exists(image_path):
                logger.error(f"Imagen no encontrada: {image_path}")
                return {
                    "success": False,
                    "description": "No se pudo analizar el paso: imagen no encontrada"
                }
            
            # Cargar imagen
            image = Image.open(image_path)
            
            # Obtener texto extraído
            extracted_text = step.get("image", {}).get("text", "")
            
            # Preparar prompt (versión reducida para evitar errores de longitud)
            prompt = self._prepare_step_prompt_short(step, context, extracted_text)
            
            # Generar descripción
            description = self._generate_description(image_path, prompt)
            
            return {
                "success": True,
                "description": description
            }
            
        except Exception as e:
            logger.error(f"Error al analizar paso: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "description": f"Error al analizar paso: {str(e)}"
            }
    
    def analyze_workflow(self, processed_data):
        """
        Analiza el flujo de trabajo completo.
        
        Args:
            processed_data: Datos procesados de la sesión
            
        Returns:
            dict: Resultados del análisis
        """
        try:
            if not self.model or not self.processor:
                logger.error("Modelo no disponible")
                return {
                    "success": False,
                    "summary": "No se pudo analizar el flujo de trabajo: modelo no disponible"
                }
            
            # Obtener contexto y tipo de flujo
            context = processed_data.get("context", "")
            workflow_type = processed_data.get("workflow_type", "general")
            
            # Obtener pasos
            steps = processed_data.get("steps", [])
            
            if not steps:
                logger.error("No hay pasos para analizar")
                return {
                    "success": False,
                    "summary": "No se pudo analizar el flujo de trabajo: no hay pasos"
                }
            
            # Seleccionar una imagen representativa (primera imagen)
            image_path = steps[0].get("image", {}).get("path", "")
            
            if not image_path or not os.path.exists(image_path):
                logger.error(f"Imagen no encontrada: {image_path}")
                return {
                    "success": False,
                    "summary": "No se pudo analizar el flujo de trabajo: imagen no encontrada"
                }
            
            # Cargar imagen
            image = Image.open(image_path)
            
            # Preparar prompt (versión reducida)
            prompt = self._prepare_workflow_prompt_short(processed_data)
            
            # Generar resumen
            summary = self._generate_description(image_path, prompt)
            
            return {
                "success": True,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Error al analizar flujo de trabajo: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "summary": f"Error al analizar flujo de trabajo: {str(e)}"
            }
    
    def _prepare_step_prompt_short(self, step, context, extracted_text):
        """
        Prepara un prompt corto para analizar un paso.
        
        Args:
            step: Datos del paso
            context: Contexto de la sesión
            extracted_text: Texto extraído de la imagen
            
        Returns:
            str: Prompt generado
        """
        prompt = f"""Eres un experto en documentación técnica. Describe esta captura de pantalla de forma clara y concisa.

Contexto: {context}
Texto extraído: {extracted_text}

Tu descripción debe:
1. Identificar la pantalla o interfaz mostrada
2. Describir los elementos principales visibles (campos, botones, etc.)
3. Explicar el propósito o función de esta pantalla
4. Ser clara y directa, evitando repeticiones
5. Usar lenguaje técnico apropiado

Descripción:"""
        
        return prompt
    
    def _prepare_step_prompt(self, step, context, extracted_text):
        """
        Prepara el prompt completo para analizar un paso.
        
        Args:
            step: Datos del paso
            context: Contexto de la sesión
            extracted_text: Texto extraído de la imagen
            
        Returns:
            str: Prompt para el modelo
        """
        # Obtener eventos
        events = step.get("events", [])
        
        # Formatear eventos (limitado)
        events_text = ""
        for i, event in enumerate(events[:5]):  # Limitar a 5 eventos
            event_type = event.get("type", "")
            if event_type == "mouse_click":
                events_text += f"- Clic en ({event.get('x', 0)}, {event.get('y', 0)})\n"
            elif event_type == "key_press":
                events_text += f"- Tecla: {event.get('key', '')}\n"
            elif event_type == "mouse_scroll":
                events_text += f"- Scroll\n"
        
        # Construir prompt
        prompt = f"""Analiza esta captura de pantalla y genera una descripción.
Paso: {step.get('index', 0) + 1}
Contexto: {context[:100]}
Eventos: {events_text}
Texto: {extracted_text[:200]}
Descripción:"""
        
        return prompt
    
    def _prepare_workflow_prompt_short(self, processed_data):
        """
        Prepara un prompt corto para analizar el flujo de trabajo.
        
        Args:
            processed_data: Datos procesados de la sesión
            
        Returns:
            str: Prompt generado
        """
        context = processed_data.get("context", "")
        workflow_type = processed_data.get("workflow_type", "general")
        steps_count = len(processed_data.get("steps", []))
        
        prompt = f"""Eres un experto en documentación técnica. Genera un resumen conciso y útil para este flujo de trabajo.

Contexto: {context}
Tipo de flujo: {workflow_type}
Número de pasos: {steps_count}

El resumen debe:
1. Explicar el propósito principal del flujo de trabajo
2. Describir brevemente qué se logrará al completarlo
3. Mencionar los puntos clave que el usuario debe entender
4. Ser claro y directo, evitando repeticiones
5. Usar lenguaje técnico apropiado

Resumen:"""
        
        return prompt
    
    def _prepare_workflow_prompt(self, processed_data):
        """
        Prepara el prompt completo para analizar el flujo de trabajo.
        
        Args:
            processed_data: Datos procesados de la sesión
            
        Returns:
            str: Prompt para el modelo
        """
        # Obtener contexto y tipo de flujo
        context = processed_data.get("context", "")
        workflow_type = processed_data.get("workflow_type", "general")
        
        # Obtener pasos
        steps = processed_data.get("steps", [])
        steps_count = len(steps)
        
        # Construir prompt
        prompt = f"""Eres un asistente experto en documentación técnica. Genera un resumen introductorio para un manual de usuario basado en la siguiente información.

Contexto del flujo de trabajo: {context[:200]}

Información general:
- Tipo de flujo de trabajo: {workflow_type}
- Número de pasos: {steps_count}

Basándote en la imagen y la información proporcionada, genera un resumen introductorio que explique el propósito general de este flujo de trabajo y qué aprenderá el usuario. El resumen debe ser claro, conciso y orientado al usuario final.

Resumen introductorio:"""
        
        return prompt
    
    def _generate_description(self, image_path, prompt, annotations=None):
        """
        Genera una descripción detallada de una imagen y sus anotaciones.
        
        Args:
            image_path: Ruta de la imagen
            prompt: Prompt base para la generación
            annotations: Lista de anotaciones (opcional)
            
        Returns:
            str: Descripción generada
        """
        try:
            # Cargar y preparar la imagen
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preparar el contexto con anotaciones
            context = self._prepare_context(prompt, annotations)
            
            # Intentar diferentes configuraciones para obtener la mejor descripción
            descriptions = []
            
            configs = [
                {'max_length': 150, 'temperature': 0.7},
                {'max_length': 200, 'temperature': 0.5},
                {'max_length': 250, 'temperature': 0.3}
            ]
            
            for config in configs:
                try:
                    # Generar descripción con configuración actual
                    inputs = self.processor(
                        images=image,
                        text=context,
                        return_tensors="pt",
                        max_length=config['max_length'],
                        padding=True,
                        truncation=True
                    ).to(self.device)
                    
                    outputs = self.model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=config['temperature'],
                        max_new_tokens=config['max_length'],
                        repetition_penalty=1.5,
                        length_penalty=1.2,
                        num_return_sequences=1
                    )
                    
                    # Decodificar y limpiar la descripción
                    description = self.processor.decode(
                        outputs[0],
                        skip_special_tokens=True
                    )
                    
                    description = self._format_description(description)
                    
                    if self._is_valid_description(description):
                        descriptions.append(description)
                    
                except Exception as e:
                    logger.warning(f"Error en configuración {config}: {str(e)}")
                    continue
            
            # Seleccionar la mejor descripción
            if descriptions:
                return self._select_best_description(descriptions)
            else:
                return self._generate_fallback_description(prompt)
            
        except Exception as e:
            logger.error(f"Error al generar descripción: {str(e)}")
            logger.error(traceback.format_exc())
            return "No se pudo generar una descripción para esta imagen."
    
    def _prepare_context(self, prompt, annotations=None):
        """
        Prepara el contexto combinando el prompt y las anotaciones.
        
        Args:
            prompt: Prompt base
            annotations: Lista de anotaciones
            
        Returns:
            str: Contexto completo
        """
        context = prompt + "\n\n"
        
        if annotations:
            context += "Eventos detectados:\n"
            for idx, ann in enumerate(annotations, 1):
                event_type = ann.get('type', '')
                x, y = ann.get('x', 0), ann.get('y', 0)
                
                if event_type == 'click':
                    context += f"{idx}. Click en posición ({x}, {y})\n"
                elif event_type == 'text':
                    text = ann.get('text', '')
                    context += f"{idx}. Texto ingresado: '{text}' en ({x}, {y})\n"
                elif event_type == 'hover':
                    context += f"{idx}. Hover en ({x}, {y})\n"
                elif event_type == 'drag':
                    end_x = ann.get('end_x', x)
                    end_y = ann.get('end_y', y)
                    context += f"{idx}. Arrastre desde ({x}, {y}) hasta ({end_x}, {end_y})\n"
        
        return context
    
    def _format_description(self, description):
        """
        Formatea y limpia una descripción generada.
        
        Args:
            description: Descripción a formatear
            
        Returns:
            str: Descripción formateada
        """
        # Eliminar espacios extra
        description = ' '.join(description.split())
        
        # Capitalizar primera letra de cada oración
        sentences = description.split('. ')
        sentences = [s.capitalize() for s in sentences if s]
        description = '. '.join(sentences)
        
        # Asegurar punto final
        if not description.endswith('.'):
            description += '.'
        
        return description
    
    def _is_valid_description(self, description):
        """
        Verifica si una descripción cumple con los criterios mínimos.
        
        Args:
            description: Descripción a validar
            
        Returns:
            bool: True si la descripción es válida
        """
        # Verificar longitud mínima (20 palabras)
        if len(description.split()) < 20:
            return False
        
        # Verificar que no sea muy repetitiva
        words = description.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Ignorar palabras cortas
                word_freq[word] = word_freq.get(word, 0) + 1
                if word_freq[word] > 3:  # Más de 3 repeticiones
                    return False
        
        return True
    
    def _select_best_description(self, descriptions):
        """
        Selecciona la mejor descripción basada en criterios de calidad.
        
        Args:
            descriptions: Lista de descripciones candidatas
            
        Returns:
            str: La mejor descripción
        """
        best_score = -1
        best_description = descriptions[0]
        
        for desc in descriptions:
            # Calcular puntuación basada en varios criterios
            words = desc.split()
            unique_words = len(set(words))
            total_words = len(words)
            avg_word_length = sum(len(w) for w in words) / total_words
            
            # Fórmula de puntuación
            score = (unique_words / total_words) * avg_word_length
            
            if score > best_score:
                best_score = score
                best_description = desc
        
        return best_description
    
    def _generate_fallback_description(self, prompt):
        """
        Genera una descripción de respaldo cuando fallan los intentos principales.
        
        Args:
            prompt: Prompt original
            
        Returns:
            str: Descripción de respaldo
        """
        return (
            "Esta imagen muestra una interfaz de usuario interactiva. "
            "Se pueden observar elementos de la interfaz y acciones del usuario "
            "que demuestran la interacción con el sistema."
        )

def create_multimodal_llm_engine(model_name="microsoft/git-base", load_in_8bit=True, load_in_4bit=False, cache_dir=None):
    """
    Crea una instancia del motor multimodal.
    
    Args:
        model_name: Nombre del modelo a utilizar
        load_in_8bit: Si se debe cargar el modelo en precisión de 8 bits
        load_in_4bit: Si se debe cargar el modelo en precisión de 4 bits
        cache_dir: Directorio de caché para modelos
        
    Returns:
        MultimodalLLMEngine: Instancia del motor multimodal
    """
    return MultimodalLLMEngine(model_name, load_in_8bit, load_in_4bit, cache_dir)
