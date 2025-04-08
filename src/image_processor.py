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
import glob
import re
import traceback

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
    
    def __init__(self, ocr_lang="spa", detect_ui_elements=True):
        """
        Inicializa el procesador de imágenes.
        
        Args:
            ocr_lang: Idioma para OCR (spa, eng, etc.)
            detect_ui_elements: Si se debe detectar elementos de UI
        """
        self.ocr_lang = ocr_lang
        self.detect_ui_elements = detect_ui_elements
        self.tesseract_available = configure_tesseract()
        
        if not self.tesseract_available:
            logger.warning("Tesseract OCR no disponible. La extracción de texto será limitada.")
        
        logger.info(f"ImageProcessor inicializado. Idioma OCR: {ocr_lang}")
        logger.info(f"Detección de elementos UI: {detect_ui_elements}")
        
        # Configuración de colores para anotaciones
        self.annotation_colors = {
            'click': (0, 0, 255),      # Rojo
            'text': (0, 255, 0),       # Verde
            'hover': (255, 165, 0),    # Naranja
            'drag': (255, 0, 255)      # Magenta
        }
        self.annotation_sizes = {
            'click': 20,
            'text': 2,
            'hover': 15,
            'drag': 3
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.line_thickness = 2
    
    def extract_text(self, image_path):
        """
        Extrae texto de una imagen utilizando técnicas avanzadas de preprocesamiento.
        
        Args:
            image_path: Ruta de la imagen
            
        Returns:
            str: Texto extraído
        """
        try:
            # Cargar imagen con OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            # Convertir a escala de grises
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Aplicar umbral adaptativo
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
            
            # Aplicar dilatación para conectar caracteres cercanos
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            dilated = cv2.dilate(binary, kernel, iterations=1)
            
            # Configurar Tesseract
            custom_config = r'--oem 3 --psm 6 -l spa+eng'
            
            # Extraer texto
            text = pytesseract.image_to_string(dilated, config=custom_config)
            
            # Limpiar texto
            text = self._clean_text(text)
            
            return text
            
        except Exception as e:
            logger.error(f"Error al extraer texto: {str(e)}")
            logger.error(traceback.format_exc())
            return ""

    def _clean_text(self, text):
        """
        Limpia el texto extraído.
        
        Args:
            text: Texto a limpiar
            
        Returns:
            str: Texto limpio
        """
        try:
            if not text:
                return ""
                
            # Reemplazar caracteres especiales comunes
            replacements = {
                '|': 'I',
                '[': '(',
                ']': ')',
                '{': '(',
                '}': ')',
                '¢': 'c',
                '©': '(c)',
                '®': '(R)',
                '°': 'º',
                '—': '-',
                '–': '-',
                '"': '"',
                '"': '"',
                ''': "'",
                ''': "'",
                '…': '...',
                '\x0c': '\n',
                'º': 'o',
                'ª': 'a',
                'ñ': 'n',
                'á': 'a',
                'é': 'e',
                'í': 'i',
                'ó': 'o',
                'ú': 'u',
                'ü': 'u',
                '¿': '?',
                '¡': '!'
            }
            
            for old, new in replacements.items():
                text = text.replace(old, new)
            
            # Eliminar caracteres no imprimibles
            text = ''.join(char for char in text if char.isprintable() or char in ['\n', '\t'])
            
            # Eliminar espacios múltiples
            text = ' '.join(text.split())
            
            # Eliminar líneas vacías múltiples
            lines = text.split('\n')
            lines = [line.strip() for line in lines]
            lines = [line for line in lines if line]
            text = '\n'.join(lines)
            
            return text
            
        except Exception as e:
            logger.error(f"Error al limpiar texto: {str(e)}")
            return text

    def process_session(self, session_path, events=None, output_path=None):
        """
        Procesa una sesión de capturas, detectando cambios significativos.
        
        Args:
            session_path: Ruta de la sesión
            events: Lista de eventos a procesar (opcional)
            output_path: Ruta de salida (opcional)
            
        Returns:
            dict: Resultados del procesamiento
        """
        try:
            # Obtener lista de capturas ordenadas por tiempo
            captures = sorted(
                glob.glob(os.path.join(session_path, "*.png")),
                key=lambda x: os.path.getmtime(x)
            )
            
            if not captures:
                return {
                    "success": False,
                    "error": f"No se encontraron capturas en: {session_path}"
                }
            
            logger.info(f"Procesando {len(captures)} capturas...")
            
            results = []
            previous_image = None
            previous_text = ""
            
            for capture_path in captures:
                try:
                    # Cargar imagen actual
                    current_image = cv2.imread(capture_path)
                    if current_image is None:
                        logger.warning(f"No se pudo cargar la imagen: {capture_path}")
                        continue
                    
                    # Obtener timestamp del archivo
                    timestamp = os.path.getmtime(capture_path)
                    
                    # Extraer texto de la imagen
                    current_text = self.extract_text(capture_path)
                    
                    if previous_image is None:
                        # Primera imagen, siempre procesar
                        logger.info(f"Procesando primera captura: {capture_path}")
                        result = self._process_capture(capture_path, events or [])
                        if result:
                            result['timestamp'] = timestamp
                            result['text'] = current_text
                            results.append(result)
                    else:
                        # Detectar cambios significativos
                        has_visual_changes = self._detect_significant_changes(previous_image, current_image)
                        has_text_changes = self._is_significant_change(current_text, previous_text)
                        
                        if has_visual_changes or has_text_changes:
                            logger.info(f"Procesando captura con cambios: {capture_path}")
                            result = self._process_capture(capture_path, events or [])
                            if result:
                                result['timestamp'] = timestamp
                                result['text'] = current_text
                                results.append(result)
                        else:
                            logger.debug(f"Omitiendo captura sin cambios significativos: {capture_path}")
                    
                    previous_image = current_image
                    previous_text = current_text
                    
                except Exception as e:
                    logger.error(f"Error procesando captura {capture_path}: {str(e)}")
                    continue
            
            logger.info(f"Procesadas {len(results)} capturas de {len(captures)} totales")
            
            # Guardar resultados si se especifica ruta
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Identificar pasos
            steps = self._identify_steps(results, events or [])
            
            # Determinar tipo de flujo de trabajo
            workflow_type = self._determine_workflow_type(steps, "")
            
            # Preparar datos procesados
            processed_data = {
                "session_path": session_path,
                "processed_at": datetime.datetime.now().isoformat(),
                "workflow_type": workflow_type,
                "steps": steps,
                "total_captures": len(captures),
                "processed_captures": len(results),
                "captures": results
            }
            
            # Guardar datos procesados
            output_file = os.path.join(session_path, "processed_data.json")
            with open(output_file, 'w') as f:
                json.dump(processed_data, f, indent=2)
            
            logger.info(f"Datos procesados guardados en: {output_file}")
            
            return {
                "success": True,
                "steps_count": len(steps),
                "workflow_type": workflow_type,
                "processed_captures": len(results)
            }
            
        except Exception as e:
            logger.error(f"Error al procesar sesión: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }

    def _process_capture(self, image_path, events):
        """
        Procesa una captura añadiendo anotaciones visuales para los eventos.
        
        Args:
            image_path: Ruta de la imagen
            events: Lista de eventos a procesar
        
        Returns:
            dict: Resultado del procesamiento
        """
        try:
            # Cargar imagen
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"No se pudo cargar la imagen: {image_path}")
                return None
            
            # Crear copia para anotaciones
            annotated = image.copy()
            
            # Procesar eventos cronológicamente
            events = sorted(events, key=lambda x: x.get('timestamp', 0))
            
            for idx, event in enumerate(events):
                event_type = event.get('type', '').lower()
                x, y = event.get('x', 0), event.get('y', 0)
                
                if event_type == 'click':
                    # Dibujar círculo para click
                    cv2.circle(
                        annotated, 
                        (x, y), 
                        self.annotation_sizes['click'], 
                        self.annotation_colors['click'], 
                        2
                    )
                    # Añadir número de secuencia
                    cv2.putText(
                        annotated,
                        str(idx + 1),
                        (x + 10, y - 10),
                        self.font,
                        self.font_scale,
                        self.annotation_colors['click'],
                        self.line_thickness
                    )
                    
                elif event_type == 'text':
                    text = event.get('text', '')
                    # Dibujar rectángulo para entrada de texto
                    rect_width = max(len(text) * 10, 100)
                    rect_height = 30
                    cv2.rectangle(
                        annotated,
                        (x - 5, y - 5),
                        (x + rect_width, y + rect_height),
                        self.annotation_colors['text'],
                        self.annotation_sizes['text']
                    )
                    # Añadir texto ingresado
                    cv2.putText(
                        annotated,
                        text[:20] + '...' if len(text) > 20 else text,
                        (x, y + 20),
                        self.font,
                        self.font_scale,
                        self.annotation_colors['text'],
                        self.line_thickness
                    )
                    
                elif event_type == 'hover':
                    # Dibujar círculo semitransparente para hover
                    overlay = annotated.copy()
                    cv2.circle(
                        overlay,
                        (x, y),
                        self.annotation_sizes['hover'],
                        self.annotation_colors['hover'],
                        -1
                    )
                    cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
                    
                elif event_type == 'drag':
                    # Obtener coordenadas de inicio y fin
                    end_x = event.get('end_x', x)
                    end_y = event.get('end_y', y)
                    # Dibujar línea de arrastre
                    cv2.line(
                        annotated,
                        (x, y),
                        (end_x, end_y),
                        self.annotation_colors['drag'],
                        self.annotation_sizes['drag']
                    )
                    # Dibujar flechas en los extremos
                    self._draw_arrow(annotated, (x, y), (end_x, end_y))
            
            # Extraer texto de la imagen
            text = self.extract_text(image_path)
            
            # Guardar imagen anotada
            output_path = image_path.replace('.png', '_annotated.png')
            cv2.imwrite(output_path, annotated)
            
            return {
                'path': output_path,
                'text': text,
                'timestamp': os.path.getmtime(image_path)
            }
            
        except Exception as e:
            logger.error(f"Error al procesar captura {image_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _draw_arrow(self, image, start_point, end_point):
        """
        Dibuja una flecha entre dos puntos.
        
        Args:
            image: Imagen donde dibujar
            start_point: Punto de inicio (x, y)
            end_point: Punto final (x, y)
        """
        try:
            # Calcular ángulo y longitud
            angle = np.arctan2(
                end_point[1] - start_point[1],
                end_point[0] - start_point[0]
            )
            arrow_length = 20
            
            # Calcular puntos de la flecha
            arrow_p1 = (
                int(end_point[0] - arrow_length * np.cos(angle + np.pi/6)),
                int(end_point[1] - arrow_length * np.sin(angle + np.pi/6))
            )
            arrow_p2 = (
                int(end_point[0] - arrow_length * np.cos(angle - np.pi/6)),
                int(end_point[1] - arrow_length * np.sin(angle - np.pi/6))
            )
            
            # Dibujar líneas de la flecha
            cv2.line(
                image,
                end_point,
                arrow_p1,
                self.annotation_colors['drag'],
                self.annotation_sizes['drag']
            )
            cv2.line(
                image,
                end_point,
                arrow_p2,
                self.annotation_colors['drag'],
                self.annotation_sizes['drag']
            )
            
        except Exception as e:
            logger.error(f"Error al dibujar flecha: {str(e)}")
    
    def _detect_significant_changes(self, prev_image, curr_image, threshold=0.001):
        """
        Detecta cambios significativos entre dos imágenes.
        
        Args:
            prev_image: Imagen anterior
            curr_image: Imagen actual
            threshold: Umbral de cambio (0-1)
            
        Returns:
            bool: True si hay cambios significativos
        """
        try:
            if prev_image is None or curr_image is None:
                logger.info("Imagen anterior o actual es None, procesando por seguridad")
                return True
                
            # Verificar dimensiones
            if prev_image.shape != curr_image.shape:
                logger.info("Las imágenes tienen dimensiones diferentes, procesando por seguridad")
                return True
                
            # Convertir a escala de grises
            prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
            
            # Aplicar suavizado para reducir ruido
            prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
            curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
            
            # Calcular diferencia absoluta
            diff = cv2.absdiff(prev_gray, curr_gray)
            
            # Aplicar umbral adaptativo
            thresholded = cv2.adaptiveThreshold(
                diff, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                11, 
                2
            )
            
            # Calcular porcentaje de cambio
            total_pixels = thresholded.size
            changed_pixels = cv2.countNonZero(thresholded)
            change_percent = changed_pixels / total_pixels
            
            # Calcular diferencia de histograma
            hist_prev = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
            hist_curr = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])
            hist_diff = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
            
            logger.info(f"Porcentaje de cambio: {change_percent:.2%}")
            logger.info(f"Diferencia de histograma: {1 - hist_diff:.2%}")
            
            # Considerar cambio si cualquiera de las métricas supera el umbral
            return change_percent > threshold or (1 - hist_diff) > threshold
            
        except Exception as e:
            logger.error(f"Error al detectar cambios: {str(e)}")
            logger.error(traceback.format_exc())
            return True  # En caso de error, procesar por seguridad

    def process_image(self, image_path):
        """
        Procesa una imagen, detectando elementos principales y difuminando datos sensibles.
        
        Args:
            image_path: Ruta de la imagen
            
        Returns:
            dict: Resultados del procesamiento
        """
        try:
            # Cargar imagen
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            # Detectar elementos de UI si está habilitado
            ui_elements = []
            if self.detect_ui_elements:
                ui_elements = self._detect_ui_elements(image)
            
            # Extraer texto
            extracted_text = self.extract_text(image_path)
            
            # Difuminar datos sensibles
            processed_image = self._blur_sensitive_regions(image)
            
            # Guardar imagen procesada
            processed_path = image_path.replace('.png', '_processed.png')
            cv2.imwrite(processed_path, processed_image)
            
            return {
                'original_path': image_path,
                'processed_path': processed_path,
                'ui_elements': ui_elements,
                'extracted_text': extracted_text
            }
            
        except Exception as e:
            logger.error(f"Error al procesar imagen: {str(e)}")
            return None
            
    def _detect_ui_elements(self, image):
        """
        Detecta elementos principales de la interfaz de usuario.
        
        Args:
            image: Imagen en formato BGR
            
        Returns:
            list: Elementos detectados
        """
        try:
            elements = []
            
            # Convertir a escala de grises
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detectar bordes
            edges = cv2.Canny(gray, 50, 150)
            
            # Detectar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Filtrar contornos pequeños
                if cv2.contourArea(contour) < 100:
                    continue
                    
                # Obtener rectángulo del contorno
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calcular relación de aspecto
                aspect_ratio = w / float(h)
                
                # Clasificar elemento según forma
                element_type = self._classify_element(image[y:y+h, x:x+w], aspect_ratio)
                
                if element_type:
                    elements.append({
                        'type': element_type,
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h
                    })
            
            return elements
            
        except Exception as e:
            logger.error(f"Error al detectar elementos UI: {str(e)}")
            return []
            
    def _classify_element(self, roi, aspect_ratio):
        """
        Clasifica un elemento de UI según su forma y contenido.
        
        Args:
            roi: Región de interés (elemento)
            aspect_ratio: Relación de aspecto del elemento
            
        Returns:
            str: Tipo de elemento o None
        """
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Detectar bordes
            edges = cv2.Canny(gray, 50, 150)
            
            # Calcular porcentaje de bordes
            edge_percent = np.count_nonzero(edges) / edges.size
            
            # Clasificar según características
            if edge_percent > 0.3 and aspect_ratio > 2:
                return 'input'  # Campo de entrada
            elif edge_percent > 0.2 and aspect_ratio < 0.5:
                return 'button'  # Botón
            elif edge_percent > 0.1 and 0.8 < aspect_ratio < 1.2:
                return 'icon'  # Icono
            elif edge_percent > 0.05:
                return 'container'  # Contenedor
                
            return None
            
        except Exception as e:
            logger.error(f"Error al clasificar elemento: {str(e)}")
            return None
            
    def _blur_sensitive_regions(self, image):
        """
        Detecta y difumina regiones con datos sensibles en la imagen.
        
        Args:
            image: Imagen en formato BGR
            
        Returns:
            numpy.ndarray: Imagen procesada
        """
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detectar texto usando Tesseract
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            
            # Procesar cada detección de texto
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if not text:
                    continue
                    
                # Verificar si el texto parece ser sensible
                if self._is_sensitive_data(text):
                    # Obtener coordenadas del texto
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    
                    # Añadir margen al área
                    margin = 5
                    x1 = max(0, x - margin)
                    y1 = max(0, y - margin)
                    x2 = min(image.shape[1], x + w + margin)
                    y2 = min(image.shape[0], y + h + margin)
                    
                    # Difuminar la región
                    roi = image[y1:y2, x1:x2]
                    blurred = cv2.GaussianBlur(roi, (15, 15), 0)
                    image[y1:y2, x1:x2] = blurred
                    
            return image
            
        except Exception as e:
            logger.error(f"Error al difuminar datos sensibles: {str(e)}")
            return image
            
    def _is_sensitive_data(self, text):
        """
        Verifica si un texto parece ser información sensible.
        
        Args:
            text: Texto a verificar
            
        Returns:
            bool: True si es información sensible
        """
        # Patrones comunes de datos sensibles
        patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Nombres completos
            r'\b\d{8,}\b',  # Números largos (posibles DNI, tarjetas, etc.)
            r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',  # Emails
            r'\b\d{3}[-.]?\d{3}[-.]?\d{3}\b',  # Teléfonos
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
                
        return False

    def _identify_steps(self, results, events):
        """
        Identifica los pasos del flujo de trabajo basándose en los resultados del procesamiento.
        
        Args:
            results: Lista de resultados de procesamiento de capturas
            events: Lista de eventos registrados
            
        Returns:
            list: Lista de pasos identificados
        """
        try:
            steps = []
            current_step = None
            
            def parse_timestamp(ts):
                """Convierte timestamp a segundos desde epoch"""
                if isinstance(ts, (int, float)):
                    return float(ts)
                try:
                    # Intentar parsear como datetime
                    dt = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                    return dt.timestamp()
                except (ValueError, TypeError):
                    return 0.0
            
            # Ordenar resultados por timestamp
            results = sorted(results, key=lambda x: parse_timestamp(x.get('timestamp', 0)))
            
            # Ordenar eventos por timestamp
            events = sorted(events, key=lambda x: parse_timestamp(x.get('timestamp', 0)))
            
            for result in results:
                if not isinstance(result, dict):
                    logger.warning(f"Resultado inválido: {result}")
                    continue
                    
                # Extraer texto de la imagen
                text = result.get('text', '')
                if not text:
                    text = self.extract_text(result.get('path', ''))
                
                # Buscar eventos asociados a esta captura
                timestamp = parse_timestamp(result.get('timestamp', 0))
                capture_events = [
                    event for event in events 
                    if abs(parse_timestamp(event.get('timestamp', 0)) - timestamp) <= 1
                ]
                
                # Determinar si hay un cambio significativo
                if not current_step or self._is_significant_change(text, current_step.get('text', '')):
                    # Crear nuevo paso
                    current_step = {
                        'text': text,
                        'events': capture_events,
                        'timestamp': timestamp,
                        'image_path': result.get('path', ''),
                        'start_time': datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    }
                    steps.append(current_step)
                    logger.info(f"Nuevo paso identificado: {len(steps)}")
                else:
                    # Actualizar paso actual
                    current_step['events'].extend(capture_events)
                    current_step['text'] = text  # Actualizar texto con la última captura
                    logger.debug(f"Actualizando paso actual: {len(steps)}")
            
            # Agregar información de fin de paso
            for i, step in enumerate(steps):
                if i < len(steps) - 1:
                    step['end_time'] = steps[i + 1]['start_time']
                else:
                    step['end_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"Identificados {len(steps)} pasos en total")
            return steps
            
        except Exception as e:
            logger.error(f"Error al identificar pasos: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def _is_significant_change(self, new_text, old_text):
        """
        Determina si hay un cambio significativo entre dos textos.
        
        Args:
            new_text: Texto nuevo
            old_text: Texto anterior
            
        Returns:
            bool: True si hay un cambio significativo
        """
        try:
            # Limpiar textos
            new_text = self._clean_text(new_text)
            old_text = self._clean_text(old_text)
            
            # Calcular similitud
            if not old_text or not new_text:
                return True
                
            # Dividir en palabras y filtrar palabras cortas
            new_words = set(word.lower() for word in new_text.split() if len(word) > 3)
            old_words = set(word.lower() for word in old_text.split() if len(word) > 3)
            
            if not new_words or not old_words:
                return True
                
            # Calcular diferencia
            diff_words = new_words.symmetric_difference(old_words)
            similarity = 1 - (len(diff_words) / max(len(new_words), len(old_words)))
            
            logger.info(f"Similitud de texto: {similarity:.2%}")
            logger.info(f"Palabras nuevas: {len(new_words - old_words)}")
            logger.info(f"Palabras eliminadas: {len(old_words - new_words)}")
            
            # Considerar cambio significativo si:
            # 1. La similitud es baja (< 0.7)
            # 2. Hay muchas palabras nuevas o eliminadas (> 5)
            return similarity < 0.7 or len(diff_words) > 5
            
        except Exception as e:
            logger.error(f"Error al comparar textos: {str(e)}")
            return True

    def _determine_workflow_type(self, steps, context):
        """
        Determina el tipo de flujo de trabajo basándose en los pasos y el contexto.
        
        Args:
            steps: Lista de pasos identificados
            context: Contexto del flujo de trabajo
            
        Returns:
            str: Tipo de flujo de trabajo
        """
        try:
            if not steps:
                return "desconocido"
            
            # Analizar eventos y texto para determinar el tipo
            event_types = set()
            for step in steps:
                for event in step.get('events', []):
                    event_types.add(event.get('type', ''))
            
            # Determinar tipo basado en eventos
            if 'click' in event_types and 'text' in event_types:
                return "formulario"
            elif 'click' in event_types and 'drag' in event_types:
                return "arrastrar_y_soltar"
            elif 'click' in event_types:
                return "navegación"
            elif 'text' in event_types:
                return "entrada_de_texto"
            else:
                return "visualización"
                
        except Exception as e:
            logger.error(f"Error al determinar tipo de flujo: {str(e)}")
            return "desconocido"

class VideoProcessor:
    """
    Procesador de vídeo para extraer frames clave y analizar secuencias.
    """
    
    def __init__(self, ocr_lang="spa"):
        """
        Inicializa el procesador de vídeo.
        
        Args:
            ocr_lang: Idioma para OCR (spa, eng, etc.)
        """
        self.ocr_lang = ocr_lang
        self.image_processor = ImageProcessor(ocr_lang)
        
        # Configurar Tesseract OCR
        self.tesseract_available = configure_tesseract()
        
        if not self.tesseract_available:
            logger.warning("Tesseract OCR no disponible. La extracción de texto será limitada.")
        
        logger.info(f"VideoProcessor inicializado. Idioma OCR: {ocr_lang}")
    
    def process_video(self, video_path, output_dir, fps=1):
        """
        Procesa un vídeo y extrae frames clave.
        
        Args:
            video_path: Ruta al archivo de vídeo
            output_dir: Directorio de salida para frames
            fps: Frames por segundo a extraer
            
        Returns:
            dict: Resultados del procesamiento
        """
        try:
            # Crear directorio de salida
            ensure_dir(output_dir)
            
            # Abrir vídeo
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"No se pudo abrir el vídeo: {video_path}")
                return {"success": False, "error": "No se pudo abrir el vídeo"}
            
            # Obtener propiedades del vídeo
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(video_fps / fps)
            
            logger.info(f"Procesando vídeo: {video_path}")
            logger.info(f"Frames totales: {total_frames}")
            logger.info(f"FPS del vídeo: {video_fps}")
            logger.info(f"Intervalo de frames: {frame_interval}")
            
            # Procesar frames
            processed_frames = []
            frame_count = 0
            previous_frame = None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extraer frame según intervalo
                if frame_count % frame_interval == 0:
                    # Detectar cambios significativos
                    if previous_frame is None or self.image_processor._detect_significant_changes(previous_frame, frame):
                        # Guardar frame
                        frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.png")
                        cv2.imwrite(frame_path, frame)
                        
                        # Procesar frame
                        processed = self.image_processor._process_capture(frame_path, [])
                        processed["frame_number"] = frame_count
                        processed["timestamp"] = frame_count / video_fps
                        processed_frames.append(processed)
                        
                        previous_frame = frame.copy()
                
                frame_count += 1
            
            # Liberar recursos
            cap.release()
            
            # Identificar pasos
            steps = self.image_processor._identify_steps(processed_frames, [])
            
            # Determinar tipo de flujo de trabajo
            workflow_type = self.image_processor._determine_workflow_type(steps, "")
            
            # Preparar datos procesados
            processed_data = {
                "video_path": video_path,
                "processed_at": datetime.datetime.now().isoformat(),
                "workflow_type": workflow_type,
                "steps": steps,
                "total_frames": total_frames,
                "processed_frames": len(processed_frames)
            }
            
            # Guardar datos procesados
            output_file = os.path.join(output_dir, "processed_data.json")
            with open(output_file, 'w') as f:
                json.dump(processed_data, f, indent=2)
            
            logger.info(f"Datos procesados guardados en: {output_file}")
            
            return {
                "success": True,
                "steps_count": len(steps),
                "workflow_type": workflow_type,
                "processed_frames": len(processed_frames)
            }
            
        except Exception as e:
            logger.error(f"Error al procesar vídeo: {str(e)}")
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}

def create_image_processor(ocr_lang="spa"):
    """
    Crea una instancia del procesador de imágenes.
    
    Args:
        ocr_lang: Idioma para OCR (spa, eng, etc.)
        
    Returns:
        ImageProcessor: Instancia del procesador
    """
    return ImageProcessor(ocr_lang)

def create_video_processor(ocr_lang="spa"):
    """
    Crea una instancia del procesador de vídeo.
    
    Args:
        ocr_lang: Idioma para OCR (spa, eng, etc.)
        
    Returns:
        VideoProcessor: Instancia del procesador
    """
    return VideoProcessor(ocr_lang)

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
