"""
Pruebas del Sistema de Documentación Automatizada

Este archivo contiene pruebas unitarias y de integración para verificar
el correcto funcionamiento del sistema.
"""

import os
import sys
import pytest
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock

# Añadir directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Importar componentes del sistema
from src.screen_capture import create_capture_manager
from src.image_processor import create_image_processor
from src.doc_generator import create_document_generator

# Configuración de pruebas
@pytest.fixture
def test_dir():
    """Crea un directorio temporal para pruebas."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def capture_manager(test_dir):
    """Crea una instancia del gestor de captura para pruebas."""
    captures_dir = os.path.join(test_dir, "captures")
    os.makedirs(captures_dir, exist_ok=True)
    return create_capture_manager(output_dir=captures_dir, interval=0.5)

@pytest.fixture
def image_processor():
    """Crea una instancia del procesador de imágenes para pruebas."""
    return create_image_processor(ocr_lang="eng")

@pytest.fixture
def doc_generator(test_dir):
    """Crea una instancia del generador de documentación para pruebas."""
    templates_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
    output_dir = os.path.join(test_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    return create_document_generator(templates_dir=templates_dir, output_dir=output_dir)

# Pruebas del módulo de captura
class TestScreenCapture:
    """Pruebas para el módulo de captura de pantalla."""
    
    def test_capture_manager_init(self, capture_manager):
        """Verifica que el gestor de captura se inicialice correctamente."""
        assert capture_manager is not None
        assert capture_manager.interval == 0.5
        assert not capture_manager.is_capturing
    
    def test_start_stop_capture(self, capture_manager):
        """Verifica que se pueda iniciar y detener una captura."""
        # Iniciar captura
        session_id = capture_manager.start_capture(context="Prueba de captura")
        assert session_id is not None
        assert capture_manager.is_capturing
        
        # Esperar un poco para que se capturen algunas imágenes
        import time
        time.sleep(1.5)
        
        # Detener captura
        result = capture_manager.stop_capture()
        assert result is not None
        assert result["session_id"] == session_id
        assert not capture_manager.is_capturing
        
        # Verificar que se crearon archivos
        session_dir = result["session_dir"]
        assert os.path.exists(session_dir)
        assert os.path.exists(os.path.join(session_dir, "events.json"))
        assert os.path.exists(os.path.join(session_dir, "metadata.json"))
        
        # Verificar que hay capturas
        captures = [f for f in os.listdir(session_dir) if f.startswith("capture_") and f.endswith(".png")]
        assert len(captures) > 0

# Pruebas del módulo de procesamiento de imágenes
class TestImageProcessor:
    """Pruebas para el módulo de procesamiento de imágenes."""
    
    def test_processor_init(self, image_processor):
        """Verifica que el procesador de imágenes se inicialice correctamente."""
        assert image_processor is not None
        assert image_processor.ocr_lang == "eng"
    
    @pytest.mark.skip(reason="Requiere una sesión de captura real")
    def test_process_session(self, image_processor, capture_manager, test_dir):
        """Verifica que se pueda procesar una sesión de captura."""
        # Crear una sesión de captura
        session_id = capture_manager.start_capture(context="Prueba de procesamiento")
        import time
        time.sleep(1.5)
        result = capture_manager.stop_capture()
        session_dir = result["session_dir"]
        
        # Procesar sesión
        process_result = image_processor.process_session(session_dir)
        assert process_result["success"]
        assert process_result["steps_count"] >= 0
        
        # Verificar que se creó el archivo de datos procesados
        assert os.path.exists(os.path.join(session_dir, "processed_data.json"))
    
    def test_extract_text_from_image(self, image_processor):
        """Verifica que se pueda extraer texto de una imagen."""
        # Crear una imagen de prueba con texto
        from PIL import Image, ImageDraw, ImageFont
        
        # Crear imagen con texto
        img = Image.new('RGB', (400, 100), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()
        
        d.text((10, 10), "Texto de prueba para OCR", fill=(0, 0, 0), font=font)
        
        # Extraer texto
        with patch('pytesseract.image_to_string', return_value="Texto de prueba para OCR"):
            text = image_processor._extract_text(img)
            assert "Texto de prueba" in text

# Pruebas del módulo de generación de documentación
class TestDocGenerator:
    """Pruebas para el módulo de generación de documentación."""
    
    def test_generator_init(self, doc_generator):
        """Verifica que el generador de documentación se inicialice correctamente."""
        assert doc_generator is not None
        assert doc_generator.jinja_env is not None
    
    def test_generate_markdown(self, doc_generator, test_dir):
        """Verifica que se pueda generar documentación en formato Markdown."""
        # Crear datos de prueba
        template_data = {
            "title": "Documento de Prueba",
            "author": "Test",
            "date": "01/01/2025",
            "context": "Contexto de prueba",
            "workflow_type": "test",
            "summary": "Resumen de prueba",
            "steps": [
                {
                    "number": 1,
                    "title": "Paso 1",
                    "description": "Descripción del paso 1",
                    "image_path": "images/test.png",
                    "events": [],
                    "pattern": "test"
                }
            ],
            "metadata": {
                "generated_at": "2025-01-01T00:00:00",
                "session_id": "test_session",
                "steps_count": 1
            }
        }
        
        # Generar Markdown
        with patch.object(doc_generator.jinja_env, 'get_template') as mock_get_template:
            mock_template = MagicMock()
            mock_template.render.return_value = "# Documento de Prueba\n\nContenido de prueba"
            mock_get_template.return_value = mock_template
            
            output_file = doc_generator._generate_markdown(template_data, test_dir)
            
            assert output_file is not None
            assert os.path.exists(output_file)
            
            with open(output_file, 'r') as f:
                content = f.read()
                assert "Documento de Prueba" in content

# Pruebas de integración
class TestIntegration:
    """Pruebas de integración entre los diferentes módulos."""
    
    @pytest.mark.skip(reason="Prueba de integración completa que requiere todos los componentes")
    def test_full_workflow(self, capture_manager, image_processor, doc_generator, test_dir):
        """Verifica el flujo de trabajo completo desde la captura hasta la generación de documentación."""
        # 1. Capturar pantalla
        session_id = capture_manager.start_capture(context="Prueba de integración")
        import time
        time.sleep(2)  # Capturar durante 2 segundos
        capture_result = capture_manager.stop_capture()
        session_dir = capture_result["session_dir"]
        
        # 2. Procesar capturas
        process_result = image_processor.process_session(session_dir)
        assert process_result["success"]
        
        # 3. Generar documentación
        # Crear datos analizados simulados
        analyzed_data = {
            "step_analyses": {
                "0": {
                    "description": "Descripción generada para el paso 1",
                    "success": True
                }
            },
            "workflow_summary": {
                "summary": "Resumen del flujo de trabajo de prueba",
                "success": True
            }
        }
        
        doc_result = doc_generator.generate_documentation(
            session_dir=session_dir,
            analyzed_data=analyzed_data,
            title="Prueba de Integración",
            author="Sistema de Pruebas",
            formats=["markdown"]
        )
        
        assert doc_result["success"]
        assert "markdown" in doc_result["outputs"]
        assert os.path.exists(doc_result["outputs"]["markdown"])

# Pruebas de la aplicación Flask
@pytest.fixture
def app():
    """Crea una instancia de la aplicación Flask para pruebas."""
    from app import app as flask_app
    flask_app.config.update({
        'TESTING': True,
        'DATA_DIR': tempfile.mkdtemp(),
    })
    
    # Crear directorios necesarios
    os.makedirs(os.path.join(flask_app.config['DATA_DIR'], 'captures'), exist_ok=True)
    os.makedirs(os.path.join(flask_app.config['DATA_DIR'], 'output'), exist_ok=True)
    
    yield flask_app
    
    # Limpiar después de las pruebas
    shutil.rmtree(flask_app.config['DATA_DIR'])

@pytest.fixture
def client(app):
    """Cliente de prueba para la aplicación Flask."""
    return app.test_client()

class TestFlaskApp:
    """Pruebas para la aplicación web Flask."""
    
    def test_index_route(self, client):
        """Verifica que la ruta principal devuelva la página de inicio."""
        response = client.get('/')
        assert response.status_code == 200
    
    def test_sessions_route(self, client):
        """Verifica que la ruta de sesiones devuelva la lista de sesiones."""
        response = client.get('/sessions')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"]
        assert "sessions" in data

if __name__ == "__main__":
    pytest.main(["-v"])
