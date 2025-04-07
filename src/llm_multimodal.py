"""
Módulo de integración de modelos multimodales adaptado para Windows

Este módulo se encarga de integrar modelos de lenguaje multimodales para
analizar capturas de pantalla y generar descripciones en lenguaje natural.
"""

import os
import json
import logging
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
    Motor de integración de modelos de lenguaje multimodales.
    """
    
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf", load_in_8bit=True, load_in_4bit=False, cache_dir=None):
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
        Carga el modelo multimodal.
        """
        try:
            logger.info(f"Cargando modelo: {self.model_name}")
            
            # Importar transformers
            from transformers import AutoProcessor
            from transformers import LlavaForConditionalGeneration
            
            # Configurar opciones de carga
            load_options = {}
            
            if self.cache_dir:
                ensure_dir(self.cache_dir)
                load_options["cache_dir"] = self.cache_dir
            
            # Verificar si CUDA está disponible
            cuda_available = torch.cuda.is_available()

            if cuda_available and self.load_in_8bit:
                load_options["load_in_8bit"] = True
                load_options["device_map"] = "auto"
            elif cuda_available and self.load_in_4bit:
                load_options["load_in_4bit"] = True
                load_options["device_map"] = "auto"
            else:
                # Sin cuantización en CPU
                load_options["device_map"] = "auto"
                # Forzar desactivación de cuantización
                self.load_in_8bit = False
                self.load_in_4bit = False
                logger.info("CUDA no disponible, desactivando cuantización")
            
            # En Windows, usar menos hilos para evitar problemas de memoria
            if self.is_windows:
                os.environ["OMP_NUM_THREADS"] = "1"
                os.environ["MKL_NUM_THREADS"] = "1"
                torch.set_num_threads(1)
            
            # Cargar procesador
            self.processor = AutoProcessor.from_pretrained(self.model_name, **load_options)
            
            # Cargar modelo
            self.model = LlavaForConditionalGeneration.from_pretrained(self.model_name, **load_options)
            
            logger.info("Modelo cargado correctamente")
            
        except Exception as e:
            logger.error(f"Error al cargar modelo: {str(e)}")
            import traceback
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
            
            # Obtener texto extraído
            extracted_text = step.get("image", {}).get("text", "")
            
            # Preparar prompt
            prompt = self._prepare_step_prompt(step, context, extracted_text)
            
            # Generar descripción
            description = self._generate_description(image, prompt)
            
            return {
                "success": True,
                "description": description
            }
            
        except Exception as e:
            logger.error(f"Error al analizar paso: {str(e)}")
            import traceback
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
            
            # Preparar prompt
            prompt = self._prepare_workflow_prompt(processed_data)
            
            # Generar resumen
            summary = self._generate_description(image, prompt)
            
            return {
                "success": True,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Error al analizar flujo de trabajo: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "summary": f"Error al analizar flujo de trabajo: {str(e)}"
            }
    
    def _prepare_step_prompt(self, step, context, extracted_text):
        """
        Prepara el prompt para analizar un paso.
        
        Args:
            step: Datos del paso
            context: Contexto de la sesión
            extracted_text: Texto extraído de la imagen
            
        Returns:
            str: Prompt para el modelo
        """
        # Obtener eventos
        events = step.get("events", [])
        
        # Formatear eventos
        events_text = ""
        for event in events:
            event_type = event.get("type", "")
            if event_type == "mouse_click":
                events_text += f"- Clic en posición ({event.get('x', 0)}, {event.get('y', 0)})\n"
            elif event_type == "key_press":
                events_text += f"- Tecla presionada: {event.get('key', '')}\n"
            elif event_type == "mouse_scroll":
                events_text += f"- Scroll en dirección ({event.get('dx', 0)}, {event.get('dy', 0)})\n"
        
        # Construir prompt
        prompt = f"""Eres un asistente experto en documentación técnica. Analiza la siguiente captura de pantalla y genera una descripción detallada de lo que está sucediendo en este paso de un flujo de trabajo.

Contexto del flujo de trabajo: {context}

Información del paso:
- Número de paso: {step.get('index', 0) + 1}
- Patrón detectado: {step.get('pattern', 'desconocido')}

Eventos registrados:
{events_text}

Texto extraído de la imagen:
{extracted_text}

Genera una descripción clara y detallada de lo que muestra la imagen y lo que está sucediendo en este paso. Explica qué elementos de la interfaz son relevantes y qué acción se está realizando. La descripción debe ser útil para un manual de usuario.

Descripción:"""
        
        return prompt
    
    def _prepare_workflow_prompt(self, processed_data):
        """
        Prepara el prompt para analizar el flujo de trabajo completo.
        
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

Contexto del flujo de trabajo: {context}

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
            # Preparar inputs
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            
            # Generar respuesta
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1
                )
            
            # Decodificar respuesta
            generated_text = self.processor.batch_decode(output, skip_special_tokens=True)[0]
            
            # Extraer solo la respuesta (eliminar el prompt)
            response = generated_text[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error al generar descripción: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return f"No se pudo generar una descripción: {str(e)}"

def create_multimodal_llm_engine(model_name="llava-hf/llava-1.5-7b-hf", load_in_8bit=True, load_in_4bit=False, cache_dir=None):
    """
    Crea una instancia del motor multimodal.
    
    Args:
        model_name: Nombre del modelo a utilizar
        load_in_8bit: Si se debe cargar el modelo en precisión de 8 bits
        load_in_4bit: Si se debe cargar el modelo en precisión de 4 bits
        cache_dir: Directorio de caché para modelos
        
    Returns:
        MultimodalLLMEngine: Instancia del motor
    """
    return MultimodalLLMEngine(model_name, load_in_8bit, load_in_4bit, cache_dir)

# Función para pruebas independientes
if __name__ == "__main__":
    import sys
    import argparse
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="Motor multimodal")
    parser.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf", help="Modelo a utilizar")
    parser.add_argument("--image", type=str, required=True, help="Ruta de la imagen a analizar")
    parser.add_argument("--prompt", type=str, default="Describe esta imagen", help="Prompt para el modelo")
    parser.add_argument("--no-8bit", action="store_true", help="Deshabilitar cuantización de 8 bits")
    parser.add_argument("--use-4bit", action="store_true", help="Usar cuantización de 4 bits")
    
    args = parser.parse_args()
    
    # Crear motor
    engine = create_multimodal_llm_engine(
        model_name=args.model,
        load_in_8bit=not args.no_8bit,
        load_in_4bit=args.use_4bit
    )
    
    # Cargar imagen
    image = Image.open(args.image)
    
    # Generar descripción
    description = engine._generate_description(image, args.prompt)
    
    # Mostrar resultado
    print("\nDescripción generada:")
    print("-" * 50)
    print(description)
    print("-" * 50)
    
    # Descargar modelo
    engine.unload_model()
