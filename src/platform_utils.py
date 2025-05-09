"""
Módulo de utilidades para compatibilidad entre plataformas

Este módulo proporciona funciones y clases para garantizar la compatibilidad
del sistema entre diferentes plataformas (Windows, Linux, macOS).
"""

import os
import sys
import platform
import logging
import subprocess
from pathlib import Path

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("platform_utils")

def get_platform():
    """
    Detecta la plataforma actual.
    
    Returns:
        str: 'windows', 'linux', 'macos' o 'unknown'
    """
    system = platform.system().lower()
    
    if system == 'windows':
        return 'windows'
    elif system == 'linux':
        return 'linux'
    elif system == 'darwin':
        return 'macos'
    else:
        return 'unknown'

def normalize_path(path):
    """
    Normaliza una ruta para la plataforma actual.
    
    Args:
        path: Ruta a normalizar
        
    Returns:
        str: Ruta normalizada
    """
    return str(Path(path))

def ensure_dir(directory):
    """
    Asegura que un directorio exista, creándolo si es necesario.
    
    Args:
        directory: Ruta del directorio
        
    Returns:
        str: Ruta normalizada del directorio
    """
    dir_path = normalize_path(directory)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def is_tesseract_available():
    """
    Verifica si Tesseract OCR está disponible en el sistema.
    
    Returns:
        bool: True si Tesseract está disponible, False en caso contrario
    """
    try:
        platform_name = get_platform()
        
        if platform_name == 'windows':
            # En Windows, verificar en PATH y en ubicaciones comunes de instalación
            try:
                # Intentar ejecutar tesseract desde PATH
                subprocess.run(["tesseract", "--version"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               check=True)
                return True
            except (subprocess.SubprocessError, FileNotFoundError):
                # Verificar ubicaciones comunes
                common_paths = [
                    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
                ]
                
                for path in common_paths:
                    if os.path.exists(path):
                        # Agregar al PATH para futuras llamadas
                        os.environ["PATH"] += os.pathsep + os.path.dirname(path)
                        return True
                
                return False
        else:
            # En Linux/macOS, simplemente intentar ejecutar el comando
            subprocess.run(["tesseract", "--version"], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE, 
                           check=True)
            return True
    except Exception as e:
        logger.warning(f"Error al verificar Tesseract: {str(e)}")
        return False

def get_tesseract_cmd():
    """
    Obtiene el comando para ejecutar Tesseract OCR.
    
    Returns:
        str: Comando de Tesseract
    """
    platform_name = get_platform()
    
    if platform_name == 'windows':
        # En Windows, buscar en ubicaciones comunes
        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        # Si no se encuentra en ubicaciones comunes, usar el comando básico
        return 'tesseract'
    else:
        # En Linux/macOS, usar el comando básico
        return 'tesseract'

def configure_tesseract():
    """
    Configura Tesseract OCR para su uso en el sistema.
    
    Returns:
        bool: True si la configuración fue exitosa, False en caso contrario
    """
    import pytesseract
    
    try:
        if get_platform() == 'windows':
            tesseract_cmd = get_tesseract_cmd()
            if tesseract_cmd != 'tesseract':
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                logger.info(f"Tesseract configurado en: {tesseract_cmd}")
            
        # Verificar que funciona
        pytesseract.get_tesseract_version()
        logger.info(f"Tesseract OCR disponible, versión: {pytesseract.get_tesseract_version()}")
        return True
    except Exception as e:
        logger.warning(f"No se pudo configurar Tesseract OCR: {str(e)}")
        return False

def get_temp_dir():
    """
    Obtiene un directorio temporal adecuado para la plataforma.
    
    Returns:
        str: Ruta al directorio temporal
    """
    import tempfile
    return tempfile.gettempdir()

def get_user_data_dir(app_name="DocAutomation"):
    """
    Obtiene el directorio de datos de usuario adecuado para la plataforma.
    
    Args:
        app_name: Nombre de la aplicación
        
    Returns:
        str: Ruta al directorio de datos de usuario
    """
    platform_name = get_platform()
    
    if platform_name == 'windows':
        base_dir = os.environ.get('APPDATA', os.path.expanduser('~'))
    elif platform_name == 'macos':
        base_dir = os.path.expanduser('~/Library/Application Support')
    else:  # linux o desconocido
        base_dir = os.environ.get('XDG_DATA_HOME', os.path.expanduser('~/.local/share'))
    
    app_dir = os.path.join(base_dir, app_name)
    os.makedirs(app_dir, exist_ok=True)
    return app_dir

def get_screenshot_backend():
    """
    Obtiene el backend adecuado para capturas de pantalla según la plataforma.
    
    Returns:
        str: Nombre del backend ('pyautogui', 'mss', etc.)
    """
    platform_name = get_platform()
    
    # PyAutoGUI funciona en todas las plataformas, pero podríamos usar
    # backends específicos si es necesario en el futuro
    return 'pyautogui'

def take_screenshot(region=None):
    """
    Toma una captura de pantalla utilizando el backend adecuado para la plataforma.
    
    Args:
        region: Región a capturar (x, y, ancho, alto) o None para pantalla completa
        
    Returns:
        PIL.Image: Imagen capturada
    """
    backend = get_screenshot_backend()
    
    if backend == 'pyautogui':
        import pyautogui
        if region:
            return pyautogui.screenshot(region=region)
        else:
            return pyautogui.screenshot()
    else:
        raise ValueError(f"Backend de captura no soportado: {backend}")

# Función para pruebas independientes
if __name__ == "__main__":
    print(f"Plataforma detectada: {get_platform()}")
    print(f"Tesseract disponible: {is_tesseract_available()}")
    print(f"Comando Tesseract: {get_tesseract_cmd()}")
    print(f"Directorio temporal: {get_temp_dir()}")
    print(f"Directorio de datos de usuario: {get_user_data_dir()}")
    print(f"Backend de captura: {get_screenshot_backend()}")
