"""
Módulo de procesamiento de imágenes actualizado para compatibilidad con Windows

Este módulo se encarga de procesar las capturas de pantalla para extraer información
relevante y preparar los datos para la generación de documentación.
"""

import os
import json
import logging
import datetime
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pathlib import Path

# Importar utilidades de plataforma
from src.platform_utils import normalize_path, ensure_dir, configure_tesseract

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("image_processor.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("image_processor")

class ImageProcessor:
    """
    Procesador de imágenes para extraer información de capturas de pantalla.
    """
    
    def __init__(self, ocr_lang="spa"):
        """
        Inicializa el procesador de imágenes.
        
        Args:
            ocr_lang: Idioma para OCR (spa, eng, etc.)
        """
        self.ocr_lang = ocr_lang
        
        # Configurar Tesseract OCR
        self.tesseract_available = configure_tesseract()
        
        if not self.tesseract_available:
            logger.warning("Tesseract OCR no disponible. La extracción de texto será limitada.")
        
        logger.info(f"ImageProcessor inicializado. Idioma OCR: {ocr_lang}")
    
    def process_session(self, session_dir):
        """
        Procesa una sesión de captura.
        
        Args:
            session_dir: Directorio de la sesión
            
        Returns:
            dict: Resultados del procesamiento
        """
        try:
            session_dir = normalize_path(session_dir)
            logger.info(f"Procesando sesión en: {session_dir}")
            
            # Verificar que el directorio existe
            if not os.path.exists(session_dir):
                logger.error(f"Directorio de sesión no encontrado: {session_dir}")
                return {"success": False, "error": "Directorio de sesión no encontrado"}
            
            # Cargar eventos
            events_file = os.path.join(session_dir, "events.json")
            if not os.path.exists(events_file):
                logger.error(f"Archivo de eventos no encontrado: {events_file}")
                return {"success": False, "error": "Archivo de eventos no encontrado"}
            
            with open(events_file, 'r') as f:
                events = json.load(f)
            
            # Cargar metadatos
            metadata_file = os.path.join(session_dir, "metadata.json")
            metadata = {}
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            # Obtener capturas
            captures = [f for f in os.listdir(session_dir) if f.startswith("capture_") and f.endswith(".png")]
            captures.sort()
            
            logger.info(f"Eventos cargados: {len(events)}")
            logger.info(f"Capturas encontradas: {len(captures)}")
            
            # Procesar capturas
            processed_captures = []
            
            for capture_file in captures:
                capture_path = os.path.join(session_dir, capture_file)
                
                # Procesar captura
                processed = self._process_capture(capture_path)
                processed["filename"] = capture_file
                
                processed_captures.append(processed)
            
            # Identificar pasos
            steps = self._identify_steps(processed_captures, events)
            
            logger.info(f"Pasos identificados: {len(steps)}")
            
            # Determinar tipo de flujo de trabajo
            workflow_type = self._determine_workflow_type(steps, metadata.get("context", ""))
            
            # Preparar datos procesados
            processed_data = {
                "session_id": os.path.basename(session_dir),
                "processed_at": datetime.datetime.now().isoformat(),
                "context": metadata.get("context", ""),
                "workflow_type": workflow_type,
                "steps": steps
            }
            
            # Guardar datos procesados
            output_file = os.path.join(session_dir, "processed_data.json")
            with open(output_file, 'w') as f:
                json.dump(processed_data, f, indent=2)
            
            logger.info(f"Datos procesados guardados en: {output_file}")
            
            return {
                "success": True,
                "steps_count": len(steps),
                "workflow_type": workflow_type
            }
            
        except Exception as e:
            logger.error(f"Error al procesar sesión: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    def _process_capture(self, image_path):
        """
        Procesa una captura de pantalla.
        
        Args:
            image_path: Ruta de la imagen
            
        Returns:
            dict: Datos procesados
        """
        try:
            # Cargar imagen
            image = Image.open(image_path)
            
            # Extraer texto
            text = self._extract_text(image)
            
            # Detectar elementos de UI
            ui_elements = self._detect_ui_elements(image)
            
            # Preparar resultado
            result = {
                "path": image_path,
                "size": image.size,
                "text": text,
                "ui_elements": ui_elements
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error al procesar captura {image_path}: {str(e)}")
            return {
                "path": image_path,
                "size": (0, 0),
                "text": "",
                "ui_elements": []
            }
    
    def _extract_text(self, image):
        """
        Extrae texto de una imagen.
        
        Args:
            image: Imagen PIL
            
        Returns:
            str: Texto extraído
        """
        if not self.tesseract_available:
            return ""
        
        try:
            # Convertir a escala de grises para mejor OCR
            gray_image = image.convert('L')
            
            # Extraer texto
            text = pytesseract.image_to_string(gray_image, lang=self.ocr_lang)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error al extraer texto: {str(e)}")
            return ""
    
    def _detect_ui_elements(self, image):
        """
        Detecta elementos de UI en una imagen.
        
        Args:
            image: Imagen PIL
            
        Returns:
            list: Elementos detectados
        """
        try:
            # Convertir a formato OpenCV
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detectar bordes
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar contornos
            min_area = 100  # Área mínima para considerar un elemento
            elements = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    elements.append({
                        "type": "unknown",
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                        "area": int(area)
                    })
            
            return elements
            
        except Exception as e:
            logger.error(f"Error al detectar elementos UI: {str(e)}")
            return []
    
    def _identify_steps(self, processed_captures, events):
        """
        Identifica pasos en la secuencia de capturas.
        
        Args:
            processed_captures: Capturas procesadas
            events: Eventos registrados
            
        Returns:
            list: Pasos identificados
        """
        steps = []
        
        # Mapear eventos a capturas
        event_map = {}
        for event in events:
            if event["type"] == "screenshot":
                event_map[event["filename"]] = []
        
        for event in events:
            if event["type"] != "screenshot":
                # Encontrar la captura más cercana anterior al evento
                event_time = datetime.datetime.fromisoformat(event["timestamp"])
                closest_capture = None
                min_diff = datetime.timedelta.max
                
                for e in events:
                    if e["type"] == "screenshot":
                        e_time = datetime.datetime.fromisoformat(e["timestamp"])
                        diff = event_time - e_time
                        if diff >= datetime.timedelta(0) and diff < min_diff:
                            min_diff = diff
                            closest_capture = e["filename"]
                
                if closest_capture and closest_capture in event_map:
                    event_map[closest_capture].append(event)
        
        # Identificar pasos significativos
        current_step_events = []
        
        for i, capture in enumerate(processed_captures):
            filename = capture["filename"]
            capture_events = event_map.get(filename, [])
            
            # Acumular eventos
            current_step_events.extend(capture_events)
            
            # Determinar si es un paso significativo
            is_significant = False
            
            # Es significativo si hay clics o pulsaciones de teclas
            has_interactions = any(e["type"] in ["mouse_click", "key_press"] for e in capture_events)
            
            # Es significativo si hay cambios significativos en el contenido
            content_changed = False
            if i > 0:
                prev_text = processed_captures[i-1]["text"]
                curr_text = capture["text"]
                
                # Calcular diferencia de texto
                if prev_text and curr_text:
                    # Diferencia simple basada en longitud
                    text_diff = abs(len(curr_text) - len(prev_text))
                    content_changed = text_diff > 50  # Umbral arbitrario
            
            is_significant = has_interactions or content_changed
            
            # Si es significativo o es la última captura, crear un paso
            if is_significant or i == len(processed_captures) - 1:
                if current_step_events or i == 0:  # Asegurar que el primer paso siempre se incluya
                    step = {
                        "index": len(steps),
                        "description": f"Paso {len(steps) + 1}",
                        "image": capture,
                        "events": current_step_events.copy(),
                        "pattern": self._detect_pattern(current_step_events)
                    }
                    
                    steps.append(step)
                    current_step_events = []
        
        return steps
    
    def _detect_pattern(self, events):
        """
        Detecta patrones en los eventos.
        
        Args:
            events: Lista de eventos
            
        Returns:
            str: Patrón detectado
        """
        # Contar tipos de eventos
        click_count = sum(1 for e in events if e["type"] == "mouse_click")
        key_count = sum(1 for e in events if e["type"] in ["key_press", "key_release"])
        scroll_count = sum(1 for e in events if e["type"] == "mouse_scroll")
        
        # Detectar patrones comunes
        if click_count > 0 and key_count > 0:
            return "input"
        elif click_count > 0 and scroll_count > 0:
            return "navigation"
        elif click_count > 0:
            return "selection"
        elif key_count > 0:
            return "typing"
        elif scroll_count > 0:
            return "scrolling"
        else:
            return "observation"
    
    def _determine_workflow_type(self, steps, context):
        """
        Determina el tipo de flujo de trabajo.
        
        Args:
            steps: Pasos identificados
            context: Contexto de la sesión
            
        Returns:
            str: Tipo de flujo de trabajo
        """
        # Palabras clave para diferentes tipos de flujos
        keywords = {
            "login": ["login", "iniciar sesión", "acceso", "usuario", "contraseña"],
            "search": ["buscar", "búsqueda", "encontrar", "filtrar"],
            "form": ["formulario", "registro", "datos", "enviar"],
            "navigation": ["navegar", "menú", "explorar"],
            "configuration": ["configurar", "ajustes", "opciones", "preferencias"]
        }
        
        # Contar ocurrencias de palabras clave en el contexto y texto de capturas
        counts = {k: 0 for k in keywords}
        
        # Verificar contexto
        context_lower = context.lower()
        for workflow_type, words in keywords.items():
            for word in words:
                if word in context_lower:
                    counts[workflow_type] += 2  # Dar más peso al contexto
        
        # Verificar texto en capturas
        for step in steps:
            text = step["image"].get("text", "").lower()
            for workflow_type, words in keywords.items():
                for word in words:
                    if word in text:
                        counts[workflow_type] += 1
        
        # Verificar patrones
        patterns = [step["pattern"] for step in steps]
        if "input" in patterns and "selection" in patterns:
            counts["form"] += 2
        if "navigation" in patterns and "selection" in patterns:
            counts["navigation"] += 2
        if "typing" in patterns and "selection" in patterns:
            counts["search"] += 2
        
        # Determinar tipo con mayor puntuación
        if any(counts.values()):
            workflow_type = max(counts.items(), key=lambda x: x[1])[0]
        else:
            workflow_type = "general"
        
        return workflow_type

def create_image_processor(ocr_lang="spa"):
    """
    Crea una instancia del procesador de imágenes.
    
    Args:
        ocr_lang: Idioma para OCR (spa, eng, etc.)
        
    Returns:
        ImageProcessor: Instancia del procesador
    """
    return ImageProcessor(ocr_lang)

# Función para pruebas independientes
if __name__ == "__main__":
    import sys
    import argparse
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="Procesador de imágenes")
    parser.add_argument("--session", type=str, required=True, help="Directorio de la sesión")
    parser.add_argument("--lang", type=str, default="spa", help="Idioma para OCR")
    
    args = parser.parse_args()
    
    # Crear procesador
    processor = create_image_processor(args.lang)
    
    # Procesar sesión
    result = processor.process_session(args.session)
    
    # Mostrar resultados
    if result["success"]:
        print(f"Procesamiento completado. Pasos identificados: {result['steps_count']}")
        print(f"Tipo de flujo de trabajo: {result['workflow_type']}")
    else:
        print(f"Error: {result['error']}")
        sys.exit(1)
