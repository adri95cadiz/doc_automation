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
                from transformers import AutoProcessor, AutoModelForCausalLM
                
                self.processor = AutoProcessor.from_pretrained(self.model_name, **load_options)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_options)
                logger.info("Modelo GIT cargado correctamente")
            elif "blip" in self.model_name.lower():
                # Cargar BLIP2 (con manejo especial para Windows)
                from transformers import AutoProcessor, BlipForConditionalGeneration
                
                self.processor = AutoProcessor.from_pretrained(self.model_name, **load_options)
                self.model = BlipForConditionalGeneration.from_pretrained(self.model_name, **load_options)
                logger.info("Modelo BLIP2 cargado correctamente")
            else:
                # Intentar cargar otros modelos multimodales
                try:
                    from transformers import AutoProcessor, AutoModelForVision2Seq
                    
                    self.processor = AutoProcessor.from_pretrained(self.model_name, **load_options)
                    self.model = AutoModelForVision2Seq.from_pretrained(self.model_name, **load_options)
                    logger.info("Modelo cargado correctamente usando AutoModelForVision2Seq")
                except Exception as e:
                    logger.error(f"Error al cargar modelo específico: {str(e)}")
                    
                    # Intentar con modelo alternativo
                    alt_model = "microsoft/git-base"
                    logger.info(f"Intentando con modelo alternativo: {alt_model}")
                    
                    from transformers import AutoProcessor, AutoModelForCausalLM
                    
                    self.processor = AutoProcessor.from_pretrained(alt_model, **load_options)
                    self.model = AutoModelForCausalLM.from_pretrained(alt_model, **load_options)
                    logger.info("Modelo alternativo cargado correctamente")
            
            logger.info("Modelo cargado correctamente")
            
        except Exception as e:
            logger.error(f"Error al cargar modelo: {str(e)}")
            logger.error(traceback.format_exc())
            self.model = None
            self.processor = None
    
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
            
            # Preparar prompt (versión reducida para evitar errores de longitud)
            prompt = self._prepare_step_prompt_short(step, context)
            
            # Generar descripción
            description = self._generate_description(image, prompt)
            
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
            summary = self._generate_description(image, prompt)
            
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
    
    def _prepare_step_prompt_short(self, step, context):
        """
        Prepara un prompt corto para analizar un paso (evita errores de longitud).
        
        Args:
            step: Datos del paso
            context: Contexto de la sesión
            
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
        
        # Construir prompt corto
        prompt = f"""Describe esta captura de pantalla. Paso {step.get('index', 0) + 1}.
Contexto: {context[:50]}
Descripción:"""
        
        return prompt
    
    def _prepare_step_prompt(self, step, context):
        """
        Prepara el prompt completo para analizar un paso.
        
        Args:
            step: Datos del paso
            context: Contexto de la sesión
            
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
Descripción:"""
        
        return prompt
    
    def _prepare_workflow_prompt_short(self, processed_data):
        """
        Prepara un prompt corto para analizar el flujo de trabajo.
        
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
        
        # Construir prompt corto
        prompt = f"""Genera un resumen para este flujo de trabajo.
Contexto: {context[:100]}
Tipo: {workflow_type}
Pasos: {steps_count}
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
    
    def _generate_description(self, image, prompt):
        """
        Genera una descripción a partir de una imagen y un prompt.
        
        Args:
            image: Imagen PIL
            prompt: Prompt para el modelo
            
        Returns:
            str: Descripción generada
        """
        try:
            # Asegurar que la imagen tenga el formato correcto
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Redimensionar imagen para mejor compatibilidad
            image = image.resize((224, 224))
            
            # Procesar imagen y texto
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            
            # Generar respuesta con límite de tokens reducido
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,  # Reducir para evitar errores
                        do_sample=False
                    )
                except RuntimeError as e:
                    if "must match the size" in str(e):
                        # Error de tamaño de tensor - usar prompt más corto
                        logger.warning("Error de tamaño de tensor, usando prompt más corto")
                        short_prompt = prompt.split("\n")[0]  # Solo la primera línea
                        inputs = self.processor(images=image, text=short_prompt, return_tensors="pt").to(self.device)
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=50,
                            do_sample=False
                        )
                    else:
                        raise e
            
            # Decodificar respuesta - Compatible con diferentes modelos
            generated_text = ""
            
            # Método 1: Usar generate_caption si está disponible (BLIP2)
            if hasattr(self.model, "generate_caption"):
                try:
                    generated_text = self.model.generate_caption(image=image, prompt=prompt[:100])  # Limitar longitud
                except Exception as caption_error:
                    logger.warning(f"Error al usar generate_caption: {str(caption_error)}")
            
            # Método 2: Usar el tokenizador del modelo
            if not generated_text:
                try:
                    if hasattr(self.model, "config"):
                        if hasattr(self.model.config, "text_config"):
                            # Para BLIP2
                            from transformers import AutoTokenizer
                            tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_config.name_or_path)
                            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        else:
                            # Para otros modelos
                            if hasattr(self.processor, "tokenizer"):
                                generated_text = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                except Exception as tokenizer_error:
                    logger.warning(f"Error al usar tokenizador: {str(tokenizer_error)}")
            
            # Método 3: Usar el procesador directamente
            if not generated_text:
                try:
                    if hasattr(self.processor, "decode"):
                        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
                except Exception as processor_error:
                    logger.warning(f"Error al usar processor.decode: {str(processor_error)}")
            
            # Extraer solo la respuesta (eliminar el prompt si está presente)
            if prompt in generated_text:
                response = generated_text[generated_text.find(prompt) + len(prompt):].strip()
            else:
                response = generated_text.strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error al generar descripción: {str(e)}")
            logger.error(traceback.format_exc())
            return f"No se pudo generar una descripción: {str(e)}"

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
