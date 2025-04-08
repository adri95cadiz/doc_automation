"""
Módulo de captura de pantalla actualizado para compatibilidad con Windows

Este módulo se encarga de capturar la pantalla y registrar eventos del usuario
durante una sesión de documentación, con soporte para múltiples plataformas.
"""

import os
import json
import time
import logging
import threading
import traceback
from datetime import datetime
import pyautogui
import cv2
import numpy as np
from pynput import mouse, keyboard
from PIL import Image
from PIL import ImageGrab
from win32api import GetSystemMetrics
import re
import pytesseract

# Importar utilidades de plataforma
from src.platform_utils import normalize_path, ensure_dir, get_platform, take_screenshot
from .image_processor import ImageProcessor

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("screen_capture.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class CaptureManager:
    """
    Gestor de captura de pantalla y eventos de usuario.
    """
    
    def __init__(self, output_dir="data/captures", interval=1.5, capture_mode="images", 
                 video_fps=30, change_threshold=0.01, min_event_interval=0.5,
                 start_delay=3, blur_sensitive_data=True):
        """
        Inicializa el gestor de captura.
        
        Args:
            output_dir: Directorio de salida para capturas
            interval: Intervalo entre capturas (segundos)
            capture_mode: Modo de captura ("images" o "video")
            video_fps: Frames por segundo para grabación de vídeo
            change_threshold: Umbral para detectar cambios significativos (0-1)
            min_event_interval: Intervalo mínimo entre eventos (segundos)
            start_delay: Delay inicial antes de comenzar la captura (segundos)
            blur_sensitive_data: Si se debe difuminar datos sensibles
        """
        self.output_dir = normalize_path(output_dir)
        self.interval = interval
        self.capture_mode = capture_mode
        self.video_fps = video_fps
        self.change_threshold = change_threshold
        self.min_event_interval = min_event_interval
        self.start_delay = start_delay
        self.blur_sensitive_data = blur_sensitive_data
        self.is_capturing = False
        self.capture_thread = None
        self.events = []
        self.session_id = None
        self.session_dir = None
        self.capture_count = 0
        self.mouse_listener = None
        self.keyboard_listener = None
        self.video_writer = None
        self.last_frame = None
        self.last_event_time = 0
        
        # Crear directorio de salida
        ensure_dir(self.output_dir)
        
        logger.info(f"CaptureManager inicializado. Directorio de salida: {self.output_dir}")
        logger.info(f"Plataforma detectada: {get_platform()}")
        logger.info(f"Modo de captura: {capture_mode}")
        if capture_mode == "video":
            logger.info(f"FPS de vídeo: {video_fps}")
        logger.info(f"Delay inicial: {start_delay} segundos")
        logger.info(f"Difuminar datos sensibles: {blur_sensitive_data}")
    
    def start_capture(self, context=""):
        """
        Inicia una nueva sesión de captura.
        
        Args:
            context: Contexto o descripción de la sesión
        """
        if self.is_capturing:
            logger.warning("Ya hay una sesión de captura en curso")
            return False

        try:
            # Generar ID de sesión con formato de fecha mejorado
            self.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.session_dir = os.path.join(self.output_dir, self.session_id)
            ensure_dir(self.session_dir)
            
            # Inicializar contadores y estado
            self.capture_count = 0
            self.events = []
            self.last_frame = None
            self.last_event_time = time.time()
            
            # Configurar video writer si es necesario
            if self.capture_mode == "video":
                video_path = os.path.join(self.session_dir, "session_recording.mp4")
                screen_size = pyautogui.size()
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    video_path, 
                    fourcc, 
                    self.video_fps,
                    (screen_size.width, screen_size.height)
                )
            
            # Iniciar listeners de eventos
            self.mouse_listener = mouse.Listener(
                on_click=self._on_click,
                on_scroll=self._on_scroll
            )
            self.keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release
            )
            
            self.mouse_listener.start()
            self.keyboard_listener.start()
            
            # Guardar metadata inicial
            metadata = {
                "session_id": self.session_id,
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "capture_mode": self.capture_mode,
                "interval": self.interval,
                "video_fps": self.video_fps if self.capture_mode == "video" else None,
                "change_threshold": self.change_threshold,
                "platform": get_platform(),
                "context": context
            }
            
            with open(os.path.join(self.session_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=4)
            
            # Esperar delay inicial
            logger.info(f"Esperando {self.start_delay} segundos antes de iniciar la captura...")
            time.sleep(self.start_delay)
            
            # Iniciar thread de captura
            self.is_capturing = True
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            logger.info(f"Sesión de captura iniciada: {self.session_id}")
            return self.session_id
            
        except Exception as e:
            logger.error(f"Error al iniciar la captura: {str(e)}")
            traceback.print_exc()
            self.is_capturing = False
            self.session_id = None
            self.session_dir = None
            return False

    def _capture_loop(self):
        """
        Bucle principal de captura de pantalla.
        """
        try:
            while self.is_capturing:
                try:
                    # Capturar pantalla usando pyautogui
                    screenshot = pyautogui.screenshot()
                    if screenshot is not None:
                        # Convertir a formato numpy para OpenCV
                        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                        
                        # Procesar frame
                        self._process_frame(frame)
                        
                    time.sleep(self.interval)
                    
                except Exception as e:
                    logger.error(f"Error en captura de pantalla: {str(e)}")
                    time.sleep(1)  # Esperar antes de reintentar
                    
        except Exception as e:
            logger.error(f"Error en el bucle de captura: {str(e)}")
            traceback.print_exc()
            self.is_capturing = False

    def stop_capture(self):
        """
        Detiene la sesión de captura actual.
        
        Returns:
            dict: Resultados de la captura
        """
        if not self.is_capturing:
            logger.warning("No hay captura en curso")
            return None
        
        try:
            # Detener captura
            self.is_capturing = False
            
            # Esperar a que termine el thread
            if self.capture_thread:
                self.capture_thread.join(timeout=5.0)
            
            # Detener grabador de vídeo si existe
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            # Detener listeners de eventos
            self._stop_event_listeners()
            
            # Guardar eventos
            if self.session_dir:
                events_file = os.path.join(self.session_dir, "events.json")
                with open(events_file, 'w') as f:
                    json.dump(self.events, f, indent=2)
                
                # Actualizar metadatos
                metadata_file = os.path.join(self.session_dir, "metadata.json")
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    metadata["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    metadata["captures_count"] = self.capture_count
                    metadata["events_count"] = len(self.events)
                    
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                except Exception as e:
                    logger.error(f"Error al actualizar metadatos: {str(e)}")
            
            logger.info(f"Captura detenida. Eventos registrados: {len(self.events)}")
            
            # Preparar resultado
            result = {
                "session_id": self.session_id,
                "session_dir": self.session_dir,
                "captures_count": self.capture_count,
                "events_count": len(self.events),
                "capture_mode": self.capture_mode
            }
            
            # Limpiar variables
            self.session_id = None
            self.session_dir = None
            self.capture_thread = None
            self.events = []
            self.capture_count = 0
            
            return result
            
        except Exception as e:
            logger.error(f"Error al detener captura: {str(e)}")
            traceback.print_exc()
            return None
    
    def _start_event_listeners(self):
        """
        Inicia los listeners de eventos de mouse y teclado.
        """
        try:
            # Listener de mouse
            self.mouse_listener = mouse.Listener(
                on_click=self._on_mouse_click,
                on_scroll=self._on_mouse_scroll
            )
            self.mouse_listener.daemon = True
            self.mouse_listener.start()
            
            # Listener de teclado
            self.keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release
            )
            self.keyboard_listener.daemon = True
            self.keyboard_listener.start()
            
            logger.debug("Listeners de eventos iniciados")
            
        except Exception as e:
            logger.error(f"Error al iniciar listeners: {str(e)}")
    
    def _stop_event_listeners(self):
        """
        Detiene los listeners de eventos.
        """
        try:
            # Detener listener de mouse
            if self.mouse_listener:
                self.mouse_listener.stop()
                self.mouse_listener = None
            
            # Detener listener de teclado
            if self.keyboard_listener:
                self.keyboard_listener.stop()
                self.keyboard_listener = None
            
            logger.debug("Listeners de eventos detenidos")
            
        except Exception as e:
            logger.error(f"Error al detener listeners: {str(e)}")
    
    def _on_click(self, x, y, button, pressed):
        """
        Callback para eventos de clic del mouse.
        
        Args:
            x: Coordenada X del clic
            y: Coordenada Y del clic
            button: Botón del mouse presionado
            pressed: True si el botón fue presionado, False si fue liberado
        """
        if not self.is_capturing:
            return
            
        try:
            event = {
                "type": "mouse_click",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "x": x,
                "y": y,
                "button": str(button),
                "pressed": pressed
            }
            
            self.events.append(event)
            logger.debug(f"Evento de mouse: {event}")
            
        except Exception as e:
            logger.error(f"Error en evento de mouse: {str(e)}")
            
    def _on_scroll(self, x, y, dx, dy):
        """
        Callback para eventos de scroll del mouse.
        
        Args:
            x: Coordenada X del scroll
            y: Coordenada Y del scroll
            dx: Desplazamiento horizontal
            dy: Desplazamiento vertical
        """
        if not self.is_capturing:
            return
            
        try:
            event = {
                "type": "mouse_scroll",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "x": x,
                "y": y,
                "dx": dx,
                "dy": dy
            }
            
            self.events.append(event)
            logger.debug(f"Evento de scroll: {event}")
            
        except Exception as e:
            logger.error(f"Error en evento de scroll: {str(e)}")
            
    def _on_key_press(self, key):
        """
        Callback para eventos de tecla presionada.
        
        Args:
            key: Tecla presionada
        """
        if not self.is_capturing:
            return
            
        try:
            event = {
                "type": "key_press",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "key": str(key)
            }
            
            self.events.append(event)
            logger.debug(f"Evento de teclado: {event}")
            
        except Exception as e:
            logger.error(f"Error en evento de teclado: {str(e)}")
            
    def _on_key_release(self, key):
        """
        Callback para eventos de tecla liberada.
        
        Args:
            key: Tecla liberada
        """
        if not self.is_capturing:
            return
            
        try:
            event = {
                "type": "key_release",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "key": str(key)
            }
            
            self.events.append(event)
            logger.debug(f"Evento de teclado: {event}")
            
        except Exception as e:
            logger.error(f"Error en evento de teclado: {str(e)}")

    def _detect_significant_change(self, current_frame, threshold=0.3):
        """
        Detecta si hay cambios significativos entre el frame actual y el último procesado.
        
        Args:
            current_frame: Frame actual en formato BGR
            threshold: Umbral de diferencia para considerar un cambio significativo (0-1)
            
        Returns:
            bool: True si hay cambios significativos, False en caso contrario
        """
        try:
            # Si es el primer frame, considerarlo como cambio significativo
            if self.last_frame is None:
                return True
                
            # Convertir frames a escala de grises
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            last_gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
            
            # Calcular diferencia absoluta
            diff = cv2.absdiff(current_gray, last_gray)
            
            # Aplicar umbral para detectar cambios significativos
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # Calcular porcentaje de píxeles diferentes
            change_percent = np.count_nonzero(thresh) / thresh.size
            
            logger.debug(f"Porcentaje de cambio detectado: {change_percent:.2%}")
            return change_percent > threshold
            
        except Exception as e:
            logger.error(f"Error detectando cambios: {str(e)}")
            # En caso de error, procesar el frame por seguridad
            return True

    def _process_frame(self, frame):
        """
        Procesa un frame según el modo de captura configurado.
        
        Args:
            frame: Frame en formato BGR a procesar
        """
        try:
            # Detectar si hay cambios significativos
            if not self._detect_significant_change(frame):
                logger.debug("No se detectaron cambios significativos, saltando frame")
                return
                
            # Actualizar último frame procesado
            self.last_frame = frame.copy()
            
            # Procesar frame para difuminar datos sensibles
            if self.blur_sensitive_data:
                frame = self._blur_sensitive_regions(frame)
            
            # Convertir BGR a RGB para guardar la imagen
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            # Generar timestamp para el nombre del archivo
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # Guardar imagen según el modo configurado
            if self.capture_mode == "images":
                # Crear directorio de sesión si no existe
                if not os.path.exists(self.session_dir):
                    os.makedirs(self.session_dir)
                    
                # Guardar imagen
                image_path = os.path.join(self.session_dir, f"capture_{timestamp}.png")
                image.save(image_path, quality=95)
                logger.info(f"Imagen guardada: {image_path}")
                self.capture_count += 1
                
            elif self.capture_mode == "video":
                # Agregar frame al video si está en modo grabación
                if self.video_writer:
                    self.video_writer.write(frame)
                    logger.debug(f"Frame agregado al video en {timestamp}")
                    
        except Exception as e:
            logger.error(f"Error procesando frame: {str(e)}")
            traceback.print_exc()

    def _blur_sensitive_regions(self, frame):
        """
        Detecta y difumina regiones con datos sensibles en la imagen.
        
        Args:
            frame: Frame en formato BGR
            
        Returns:
            numpy.ndarray: Frame procesado
        """
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
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
                    x2 = min(frame.shape[1], x + w + margin)
                    y2 = min(frame.shape[0], y + h + margin)
                    
                    # Difuminar la región
                    roi = frame[y1:y2, x1:x2]
                    blurred = cv2.GaussianBlur(roi, (15, 15), 0)
                    frame[y1:y2, x1:x2] = blurred
                    
            return frame
            
        except Exception as e:
            logger.error(f"Error al difuminar datos sensibles: {str(e)}")
            return frame
            
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

class ScreenRecorder:
    def __init__(self, output_dir=None, fps=10, quality=80):
        """
        Inicializa el grabador de pantalla.
        
        Args:
            output_dir: Directorio de salida
            fps: Frames por segundo
            quality: Calidad de compresión (1-100)
        """
        self.output_dir = output_dir or os.path.join(os.getcwd(), "captures")
        self.fps = fps
        self.quality = quality
        self.recording = False
        self.frame_buffer = []
        self.buffer_size = 30  # Mantener últimos 30 frames
        
        # Configurar codec y formato de video
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.frame_size = None
        
    def start_recording(self):
        """
        Inicia la grabación de video.
        """
        if self.capture_mode != "video":
            raise ValueError("El modo de captura debe ser 'video' para iniciar la grabación")
            
        if self.recording:
            logger.warning("La grabación ya está en curso")
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_path = os.path.join(self.output_dir, f"recording_{timestamp}.mp4")
            
            # Obtener dimensiones de la pantalla
            screen_width = GetSystemMetrics(0)
            screen_height = GetSystemMetrics(1)
            
            # Configurar el writer de video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.video_path,
                fourcc,
                30.0, # fps
                (screen_width, screen_height)
            )
            
            self.recording = True
            logger.info(f"Iniciando grabación en: {self.video_path}")
            
        except Exception as e:
            logger.error(f"Error al iniciar la grabación: {str(e)}")
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                
    def stop_recording(self):
        """
        Detiene la grabación de video actual.
        """
        if not self.recording:
            logger.warning("No hay grabación en curso")
            return
            
        try:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                
            self.recording = False
            
            # Procesar el video si hay un procesador disponible
            if self.event_processor:
                self.event_processor.process_event({
                    'type': 'recording',
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'path': self.video_path
                })
                
            logger.info(f"Grabación detenida y guardada en: {self.video_path}")
            
        except Exception as e:
            logger.error(f"Error al detener la grabación: {str(e)}")
        finally:
            self.recording = False
            self.video_path = None

class VideoProcessor:
    """
    Clase para procesar videos y extraer frames clave.
    """
    
    def __init__(self, change_threshold=0.3):
        """
        Inicializa el procesador de video.
        
        Args:
            change_threshold: Umbral para detectar cambios significativos (0-1)
        """
        self.change_threshold = change_threshold
        
    def extract_keyframes(self, video_path, output_dir):
        """
        Extrae frames clave de un video basado en cambios significativos.
        
        Args:
            video_path: Ruta al archivo de video
            output_dir: Directorio donde guardar los frames
            
        Returns:
            list: Lista de rutas a los frames extraídos
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"No se pudo abrir el video: {video_path}")
                
            frame_paths = []
            prev_frame = None
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Procesar primer frame
                if prev_frame is None:
                    frame_path = self._save_frame(frame, output_dir, frame_count)
                    frame_paths.append(frame_path)
                    prev_frame = frame
                    frame_count += 1
                    continue
                
                # Detectar cambios significativos
                if self._detect_significant_changes(prev_frame, frame):
                    frame_path = self._save_frame(frame, output_dir, frame_count)
                    frame_paths.append(frame_path)
                    prev_frame = frame
                
                frame_count += 1
                
            cap.release()
            logger.info(f"Extraídos {len(frame_paths)} frames clave de {frame_count} frames totales")
            return frame_paths
            
        except Exception as e:
            logger.error(f"Error al procesar video: {str(e)}")
            return []
            
    def _detect_significant_changes(self, prev_frame, curr_frame):
        """
        Detecta si hay cambios significativos entre dos frames.
        
        Args:
            prev_frame: Frame anterior
            curr_frame: Frame actual
            
        Returns:
            bool: True si hay cambios significativos
        """
        try:
            # Convertir a escala de grises
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Calcular diferencia absoluta
            diff = cv2.absdiff(prev_gray, curr_gray)
            
            # Aplicar umbral
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # Calcular porcentaje de cambio
            change_percent = np.count_nonzero(thresh) / thresh.size
            
            return change_percent > self.change_threshold
            
        except Exception as e:
            logger.error(f"Error al detectar cambios: {str(e)}")
            return False
            
    def _save_frame(self, frame, output_dir, frame_count):
        """
        Guarda un frame como imagen.
        
        Args:
            frame: Frame a guardar
            output_dir: Directorio de salida
            frame_count: Número de frame
            
        Returns:
            str: Ruta al archivo guardado
        """
        try:
            filename = f"frame_{frame_count:04d}.png"
            filepath = os.path.join(output_dir, filename)
            
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_PNG_COMPRESSION, 95])
            return filepath
            
        except Exception as e:
            logger.error(f"Error al guardar frame: {str(e)}")
            return None

class ImageProcessor:
    """Clase para procesar imágenes y detectar cambios significativos."""
    
    def __init__(self):
        """Inicializa el procesador de imágenes."""
        self.last_image = None
        
    def process_image(self, image):
        """
        Procesa una imagen y detecta si hay cambios significativos.
        
        Args:
            image (numpy.ndarray): Imagen a procesar
            
        Returns:
            bool: True si hay cambios significativos
        """
        if self.last_image is None:
            self.last_image = image
            return True
            
        try:
            # Convertir a escala de grises
            gray1 = cv2.cvtColor(self.last_image, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calcular diferencia
            diff = cv2.absdiff(gray1, gray2)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # Calcular porcentaje de cambio
            change_percent = np.count_nonzero(thresh) / thresh.size
            
            # Actualizar última imagen
            self.last_image = image
            
            return change_percent > 0.3  # 30% de cambio mínimo
            
        except Exception as e:
            logger.error(f"Error al procesar imagen: {str(e)}")
            return True  # Por defecto procesar si hay error

class VideoProcessor:
    """Clase para grabar y procesar video de la pantalla."""
    
    def __init__(self, image_processor, frame_interval=30, change_threshold=0.3):
        """
        Inicializa el procesador de video.
        
        Args:
            image_processor (ImageProcessor): Procesador de imágenes
            frame_interval (int): Intervalo de frames para procesar
            change_threshold (float): Umbral de cambio para detectar eventos
        """
        self.image_processor = image_processor
        self.frame_interval = frame_interval
        self.change_threshold = change_threshold
        self.writer = None
        self.is_recording = False
        self.frame_count = 0
        
    def start_recording(self, output_path):
        """
        Inicia la grabación de video.
        
        Args:
            output_path (str): Ruta donde guardar el video
        """
        try:
            # Obtener dimensiones de la pantalla
            screen = pyautogui.screenshot()
            height, width = screen.size
            
            # Configurar writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(
                output_path, 
                fourcc, 
                30.0, 
                (width, height)
            )
            
            self.is_recording = True
            logger.info(f"Iniciando grabación en {output_path}")
            
        except Exception as e:
            logger.error(f"Error al iniciar grabación: {str(e)}")
            raise
            
    def stop_recording(self):
        """Detiene la grabación de video."""
        if self.writer:
            self.writer.release()
        self.is_recording = False
        logger.info("Grabación detenida")
        
    def process_frame(self, frame):
        """
        Procesa un frame de video.
        
        Args:
            frame (numpy.ndarray): Frame a procesar
            
        Returns:
            bool: True si el frame debe ser procesado
        """
        self.frame_count += 1
        
        # Procesar cada N frames
        if self.frame_count % self.frame_interval == 0:
            return self.image_processor.process_image(frame)
            
        return False

def create_capture_manager(output_dir="captures", interval=2.0, capture_mode="images"):
    """
    Crea y configura el gestor de captura.
    
    Args:
        output_dir (str): Directorio donde se guardarán las capturas
        interval (float): Intervalo entre capturas en segundos
        capture_mode (str): Modo de captura ('images' o 'video')
        
    Returns:
        dict: Configuración del gestor de captura
    """
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    return {
        'output_dir': output_dir,
        'interval': interval,
        'capture_mode': capture_mode,
        'is_running': False,
        'session_id': None,
        'captures_count': 0,
        'events_count': 0
    }

def _capture_loop(config):
    """
    Ejecuta el bucle principal de captura.
    
    Args:
        config (dict): Configuración del gestor de captura
    """
    session_id = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(config['output_dir'], session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    config['is_running'] = True
    config['session_id'] = session_id
    
    try:
        if config['capture_mode'] == 'video':
            # Configurar captura de video
            video_processor = VideoProcessor(
                image_processor=ImageProcessor(),
                frame_interval=30,  # Procesar cada 30 frames
                change_threshold=0.3  # 30% de cambio mínimo
            )
            
            # Iniciar grabación de video
            video_path = os.path.join(session_dir, f"session_{session_id}.avi")
            video_processor.start_recording(video_path)
            
            while config['is_running']:
                time.sleep(config['interval'])
                
        else:  # Modo imágenes
            while config['is_running']:
                try:
                    # Capturar pantalla
                    screenshot = pyautogui.screenshot()
                    
                    # Guardar captura
                    capture_path = os.path.join(
                        session_dir, 
                        f"capture_{time.strftime('%Y%m%d_%H%M%S')}.png"
                    )
                    screenshot.save(capture_path)
                    
                    config['captures_count'] += 1
                    logger.debug(f"Captura guardada: {capture_path}")
                    
                    # Esperar intervalo
                    time.sleep(config['interval'])
                    
                except Exception as e:
                    logger.error(f"Error en captura: {str(e)}")
                    time.sleep(1)  # Esperar antes de reintentar
                    
    except Exception as e:
        logger.error(f"Error en bucle de captura: {str(e)}")
        raise
    finally:
        config['is_running'] = False
        if config['capture_mode'] == 'video':
            video_processor.stop_recording()

def start_capture(config):
    """
    Inicia la captura de pantalla.
    
    Args:
        config: Configuración del gestor de capturas
    """
    try:
        if config['running']:
            logger.warning("La captura ya está en ejecución")
            return
            
        config['running'] = True
        
        if config['capture_mode'] == 'video':
            config['recorder'].start_recording()
        else:
            config['capture_thread'] = threading.Thread(
                target=_capture_loop,
                args=(config,)
            )
            config['capture_thread'].start()
            
        logger.info(f"Captura iniciada en modo {config['capture_mode']}")
        
    except Exception as e:
        logger.error(f"Error al iniciar captura: {str(e)}")
        config['running'] = False
        raise

def stop_capture(self):
    """
    Detiene la sesión de captura actual.
    
    Returns:
        dict: Resultados de la captura
    """
    if not self.is_capturing:
        logger.warning("No hay captura en curso")
        return None
    
    try:
        # Detener captura
        self.is_capturing = False
        
        # Esperar a que termine el thread
        if self.capture_thread:
            self.capture_thread.join(timeout=5.0)
        
        # Detener grabador de vídeo si existe
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        # Detener listeners de eventos
        self._stop_event_listeners()
        
        # Guardar eventos
        if self.session_dir:
            events_file = os.path.join(self.session_dir, "events.json")
            with open(events_file, 'w') as f:
                json.dump(self.events, f, indent=2)
            
            # Actualizar metadatos
            metadata_file = os.path.join(self.session_dir, "metadata.json")
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                metadata["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                metadata["captures_count"] = self.capture_count
                metadata["events_count"] = len(self.events)
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                logger.error(f"Error al actualizar metadatos: {str(e)}")
        
        logger.info(f"Captura detenida. Eventos registrados: {len(self.events)}")
        
        # Preparar resultado
        result = {
            "session_id": self.session_id,
            "session_dir": self.session_dir,
            "captures_count": self.capture_count,
            "events_count": len(self.events),
            "capture_mode": self.capture_mode
        }
        
        # Limpiar variables
        self.session_id = None
        self.session_dir = None
        self.capture_thread = None
        self.events = []
        self.capture_count = 0
        
        return result
        
    except Exception as e:
        logger.error(f"Error al detener captura: {str(e)}")
        traceback.print_exc()
        return None

def get_last_capture(config):
    """
    Obtiene información de la última captura.
    
    Args:
        config: Configuración del gestor de capturas
        
    Returns:
        dict: Información de la última captura o None
    """
    return config.get('last_capture')

def process_video(video_path, output_dir, frame_interval=30):
    """
    Procesa un video para extraer frames relevantes.
    
    Args:
        video_path: Ruta al archivo de video
        output_dir: Directorio para guardar frames
        frame_interval: Intervalo entre frames a procesar
        
    Returns:
        list: Rutas de los frames extraídos
    """
    try:
        processor = VideoProcessor(
            image_processor=ImageProcessor(),
            frame_interval=frame_interval
        )
        return processor.process_video(video_path, output_dir)
        
    except Exception as e:
        logger.error(f"Error al procesar video: {str(e)}")
        raise

def main():
    """
    Función principal que ejecuta la captura de pantalla.
    """
    try:
        # Configuración por defecto
        config = create_capture_manager(
            output_dir="captures",
            interval=2.0,
            capture_mode="images"
        )
        
        logger.info("Iniciando captura de pantalla...")
        logger.info(f"Modo: {config['capture_mode']}")
        logger.info(f"Directorio de salida: {config['output_dir']}")
        logger.info("Presiona Ctrl+C para detener la captura")
        
        # Iniciar bucle de captura
        _capture_loop(config)
        
    except KeyboardInterrupt:
        logger.info("\nCaptura detenida por el usuario")
    except Exception as e:
        logger.error(f"Error durante la captura: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("Finalizando captura...")

if __name__ == "__main__":
    main()
