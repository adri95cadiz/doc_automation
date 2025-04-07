"""
Módulo de captura de pantalla actualizado para compatibilidad con Windows

Este módulo se encarga de capturar la pantalla y registrar eventos del usuario
durante una sesión de documentación, con soporte para múltiples plataformas.
"""

import os
import json
import time
import logging
import datetime
import threading
from pathlib import Path
import pyautogui
from pynput import mouse, keyboard
from PIL import Image

# Importar utilidades de plataforma
from src.platform_utils import normalize_path, ensure_dir, get_platform, take_screenshot

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("screen_capture.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("screen_capture")

class CaptureManager:
    """
    Gestor de captura de pantalla y eventos de usuario.
    """
    
    def __init__(self, output_dir="data/captures", interval=1.5):
        """
        Inicializa el gestor de captura.
        
        Args:
            output_dir: Directorio de salida para capturas
            interval: Intervalo entre capturas (segundos)
        """
        self.output_dir = normalize_path(output_dir)
        self.interval = interval
        self.is_capturing = False
        self.capture_thread = None
        self.events = []
        self.session_id = None
        self.session_dir = None
        self.capture_count = 0
        self.mouse_listener = None
        self.keyboard_listener = None
        
        # Crear directorio de salida
        ensure_dir(self.output_dir)
        
        logger.info(f"CaptureManager inicializado. Directorio de salida: {self.output_dir}")
        logger.info(f"Plataforma detectada: {get_platform()}")
    
    def start_capture(self, context=""):
        """
        Inicia una sesión de captura.
        
        Args:
            context: Contexto de la sesión
            
        Returns:
            str: ID de la sesión
        """
        if self.is_capturing:
            logger.warning("Ya hay una captura en curso")
            return self.session_id
        
        # Generar ID de sesión
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = timestamp
        
        # Crear directorio para la sesión
        self.session_dir = os.path.join(self.output_dir, self.session_id)
        ensure_dir(self.session_dir)
        
        # Inicializar variables
        self.events = []
        self.capture_count = 0
        
        # Guardar metadatos
        metadata = {
            "session_id": self.session_id,
            "start_time": datetime.datetime.now().isoformat(),
            "context": context,
            "platform": get_platform(),
            "screen_size": pyautogui.size()
        }
        
        with open(os.path.join(self.session_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Iniciar captura
        self.is_capturing = True
        
        # Iniciar listeners de eventos
        self._start_event_listeners()
        
        # Iniciar thread de captura
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        logger.info(f"Captura iniciada. ID de sesión: {self.session_id}")
        
        return self.session_id
    
    def stop_capture(self):
        """
        Detiene la sesión de captura actual.
        
        Returns:
            dict: Resultados de la captura
        """
        if not self.is_capturing:
            logger.warning("No hay captura en curso")
            return None
        
        # Detener captura
        self.is_capturing = False
        
        # Esperar a que termine el thread
        if self.capture_thread:
            self.capture_thread.join(timeout=5.0)
        
        # Detener listeners de eventos
        self._stop_event_listeners()
        
        # Guardar eventos
        events_file = os.path.join(self.session_dir, "events.json")
        with open(events_file, 'w') as f:
            json.dump(self.events, f, indent=2)
        
        # Actualizar metadatos
        metadata_file = os.path.join(self.session_dir, "metadata.json")
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            metadata["end_time"] = datetime.datetime.now().isoformat()
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
            "events_count": len(self.events)
        }
        
        # Limpiar variables
        session_id = self.session_id
        session_dir = self.session_dir
        
        self.session_id = None
        self.session_dir = None
        self.capture_thread = None
        
        return result
    
    def _capture_loop(self):
        """
        Bucle principal de captura.
        """
        try:
            while self.is_capturing:
                # Tomar captura
                self._take_screenshot()
                
                # Esperar intervalo
                time.sleep(self.interval)
        except Exception as e:
            logger.error(f"Error en bucle de captura: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.is_capturing = False
    
    def _take_screenshot(self):
        """
        Toma una captura de pantalla.
        """
        try:
            # Tomar captura usando la función de plataforma
            screenshot = take_screenshot()
            
            # Guardar captura
            filename = f"capture_{self.capture_count:04d}.png"
            filepath = os.path.join(self.session_dir, filename)
            screenshot.save(filepath)
            
            # Registrar evento
            event = {
                "type": "screenshot",
                "timestamp": datetime.datetime.now().isoformat(),
                "filename": filename,
                "index": self.capture_count
            }
            
            self.events.append(event)
            self.capture_count += 1
            
            logger.debug(f"Captura guardada: {filepath}")
            
        except Exception as e:
            logger.error(f"Error al tomar captura: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
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
    
    def _on_mouse_click(self, x, y, button, pressed):
        """
        Callback para eventos de clic de mouse.
        """
        if not self.is_capturing:
            return
        
        try:
            if pressed:
                event = {
                    "type": "mouse_click",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "x": x,
                    "y": y,
                    "button": str(button)
                }
                
                self.events.append(event)
                logger.debug(f"Evento de clic: {x}, {y}, {button}")
                
                # Tomar captura inmediata en clics
                self._take_screenshot()
                
        except Exception as e:
            logger.error(f"Error en evento de mouse: {str(e)}")
    
    def _on_mouse_scroll(self, x, y, dx, dy):
        """
        Callback para eventos de scroll de mouse.
        """
        if not self.is_capturing:
            return
        
        try:
            event = {
                "type": "mouse_scroll",
                "timestamp": datetime.datetime.now().isoformat(),
                "x": x,
                "y": y,
                "dx": dx,
                "dy": dy
            }
            
            self.events.append(event)
            logger.debug(f"Evento de scroll: {x}, {y}, {dx}, {dy}")
            
        except Exception as e:
            logger.error(f"Error en evento de scroll: {str(e)}")
    
    def _on_key_press(self, key):
        """
        Callback para eventos de tecla presionada.
        """
        if not self.is_capturing:
            return
        
        try:
            # Convertir key a string de forma segura
            key_str = str(key)
            if hasattr(key, 'char') and key.char:
                key_str = key.char
            
            event = {
                "type": "key_press",
                "timestamp": datetime.datetime.now().isoformat(),
                "key": key_str
            }
            
            self.events.append(event)
            logger.debug(f"Tecla presionada: {key_str}")
            
        except Exception as e:
            logger.error(f"Error en evento de teclado: {str(e)}")
    
    def _on_key_release(self, key):
        """
        Callback para eventos de tecla liberada.
        """
        if not self.is_capturing:
            return
        
        try:
            # Convertir key a string de forma segura
            key_str = str(key)
            if hasattr(key, 'char') and key.char:
                key_str = key.char
            
            event = {
                "type": "key_release",
                "timestamp": datetime.datetime.now().isoformat(),
                "key": key_str
            }
            
            self.events.append(event)
            logger.debug(f"Tecla liberada: {key_str}")
            
            # Tomar captura después de Enter o Tab
            if key == keyboard.Key.enter or key == keyboard.Key.tab:
                self._take_screenshot()
            
        except Exception as e:
            logger.error(f"Error en evento de teclado: {str(e)}")

def create_capture_manager(output_dir="data/captures", interval=1.5):
    """
    Crea una instancia del gestor de captura.
    
    Args:
        output_dir: Directorio de salida para capturas
        interval: Intervalo entre capturas (segundos)
        
    Returns:
        CaptureManager: Instancia del gestor
    """
    return CaptureManager(output_dir, interval)

# Función para pruebas independientes
if __name__ == "__main__":
    import sys
    import argparse
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="Gestor de captura de pantalla")
    parser.add_argument("--output", type=str, default="data/captures", help="Directorio de salida")
    parser.add_argument("--interval", type=float, default=1.5, help="Intervalo entre capturas (segundos)")
    parser.add_argument("--duration", type=int, default=10, help="Duración de la captura (segundos)")
    
    args = parser.parse_args()
    
    # Crear gestor
    manager = create_capture_manager(args.output, args.interval)
    
    # Iniciar captura
    print(f"Iniciando captura de prueba ({args.duration} segundos)...")
    session_id = manager.start_capture(context="Prueba desde línea de comandos")
    
    # Esperar duración
    try:
        for i in range(args.duration):
            sys.stdout.write(f"\rCapturando... {i+1}/{args.duration} segundos")
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nCaptura interrumpida por usuario")
    
    # Detener captura
    print("\nDeteniendo captura...")
    result = manager.stop_capture()
    
    # Mostrar resultados
    print(f"Captura finalizada. ID de sesión: {result['session_id']}")
    print(f"Capturas realizadas: {result['captures_count']}")
    print(f"Eventos registrados: {result['events_count']}")
    print(f"Directorio de sesión: {result['session_dir']}")
