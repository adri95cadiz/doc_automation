"""
Aplicación principal del Sistema de Documentación Automatizada

Este archivo implementa la aplicación web Flask que integra todos los componentes
del sistema y proporciona una interfaz para su uso.
"""

import os
import json
import logging
import datetime
import argparse
from flask import Flask, render_template, request, jsonify, send_from_directory
from src.screen_capture import create_capture_manager
from src.image_processor import create_image_processor
from src.llm_multimodal import create_multimodal_llm_engine
from src.doc_generator import create_document_generator

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("doc_automation")

# Crear aplicación Flask
app = Flask(__name__)

# Configuración
app.config.update(
    SECRET_KEY='clave-secreta-desarrollo',
    DATA_DIR=os.path.join(os.path.dirname(__file__), 'data'),
    CAPTURES_DIR=os.path.join(os.path.dirname(__file__), 'data', 'captures'),
    OUTPUT_DIR=os.path.join(os.path.dirname(__file__), 'data', 'output'),
    TEMPLATES_DIR=os.path.join(os.path.dirname(__file__), 'templates'),
    OCR_LANG='spa',
    LLM_MODEL='llava-hf/llava-1.5-7b-hf',
    LOAD_IN_8BIT=True
)

# Crear directorios necesarios
os.makedirs(app.config['DATA_DIR'], exist_ok=True)
os.makedirs(app.config['CAPTURES_DIR'], exist_ok=True)
os.makedirs(app.config['OUTPUT_DIR'], exist_ok=True)

# Inicializar componentes
capture_manager = None
image_processor = None
llm_engine = None
doc_generator = None

def init_components(args):
    """Inicializa los componentes del sistema."""
    global capture_manager, image_processor, llm_engine, doc_generator
    
    logger.info("Inicializando componentes del sistema...")
    
    # Inicializar gestor de captura
    capture_manager = create_capture_manager(
        output_dir=app.config['CAPTURES_DIR'],
        interval=1.5
    )
    logger.info("Gestor de captura inicializado")
    
    # Inicializar procesador de imágenes
    image_processor = create_image_processor(
        ocr_lang=app.config['OCR_LANG']
    )
    logger.info("Procesador de imágenes inicializado")
    
    # Inicializar motor multimodal si no está deshabilitado
    if not args.no_multimodal:
        try:
            logger.info(f"Inicializando motor multimodal con modelo: {app.config['LLM_MODEL']}")
            llm_engine = create_multimodal_llm_engine(
                model_name=app.config['LLM_MODEL'],
                load_in_8bit=app.config['LOAD_IN_8BIT'],
                cache_dir=os.path.join(app.config['DATA_DIR'], 'models')
            )
            logger.info("Motor multimodal inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar motor multimodal: {str(e)}")
            logger.warning("El sistema funcionará sin capacidades multimodales")
    else:
        logger.info("Motor multimodal deshabilitado por argumento --no-multimodal")
    
    # Inicializar generador de documentación
    doc_generator = create_document_generator(
        templates_dir=app.config['TEMPLATES_DIR'],
        output_dir=app.config['OUTPUT_DIR']
    )
    logger.info("Generador de documentación inicializado")
    
    logger.info("Todos los componentes inicializados correctamente")

# Rutas de la aplicación
@app.route('/')
def index():
    """Página principal."""
    return render_template('index.html')

@app.route('/start_capture', methods=['POST'])
def start_capture():
    """Inicia una sesión de captura."""
    try:
        # Obtener parámetros
        title = request.form.get('title', 'Sesión sin título')
        context = request.form.get('context', '')
        interval = float(request.form.get('interval', 1.5))
        
        # Validar intervalo
        if interval < 0.5 or interval > 10:
            return jsonify({
                'success': False,
                'error': 'El intervalo debe estar entre 0.5 y 10 segundos'
            }), 400
        
        # Actualizar intervalo de captura
        capture_manager.interval = interval
        
        # Iniciar captura
        session_id = capture_manager.start_capture(context=context)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': f'Captura iniciada con ID: {session_id}'
        })
    except Exception as e:
        logger.error(f"Error al iniciar captura: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    """Detiene la sesión de captura actual."""
    try:
        # Detener captura
        result = capture_manager.stop_capture()
        
        if not result:
            return jsonify({
                'success': False,
                'error': 'No hay captura en curso'
            }), 400
        
        return jsonify({
            'success': True,
            'session_id': result['session_id'],
            'events_count': result['events_count'],
            'message': f'Captura detenida. {result["events_count"]} eventos registrados.'
        })
    except Exception as e:
        logger.error(f"Error al detener captura: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/process_session/<session_id>', methods=['POST'])
def process_session(session_id):
    """Procesa una sesión de captura."""
    try:
        # Verificar que la sesión existe
        session_dir = os.path.join(app.config['CAPTURES_DIR'], session_id)
        if not os.path.exists(session_dir):
            return jsonify({
                'success': False,
                'error': f'Sesión no encontrada: {session_id}'
            }), 404
        
        # Procesar sesión
        logger.info(f"Procesando sesión: {session_id}")
        result = image_processor.process_session(session_dir)
        
        if not result['success']:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Error desconocido al procesar sesión')
            }), 500
        
        # Si hay motor multimodal, analizar con él
        analyzed_data = {'step_analyses': {}, 'workflow_summary': {}}
        
        if llm_engine:
            try:
                # Cargar datos procesados
                processed_file = os.path.join(session_dir, "processed_data.json")
                with open(processed_file, 'r') as f:
                    processed_data = json.load(f)
                
                # Analizar cada paso
                logger.info(f"Analizando pasos con modelo multimodal...")
                for i, step in enumerate(processed_data.get('steps', [])):
                    logger.info(f"Analizando paso {i+1}/{len(processed_data.get('steps', []))}")
                    step_analysis = llm_engine.analyze_step(
                        step, 
                        context=processed_data.get('context', '')
                    )
                    analyzed_data['step_analyses'][str(i)] = step_analysis
                
                # Analizar flujo completo
                logger.info(f"Generando resumen del flujo de trabajo...")
                workflow_summary = llm_engine.analyze_workflow(processed_data)
                analyzed_data['workflow_summary'] = workflow_summary
                
                # Guardar análisis
                analysis_file = os.path.join(session_dir, "analyzed_data.json")
                with open(analysis_file, 'w') as f:
                    json.dump(analyzed_data, f, indent=2)
                
                logger.info(f"Análisis multimodal completado y guardado en: {analysis_file}")
            except Exception as e:
                logger.error(f"Error en análisis multimodal: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning("No hay motor multimodal disponible, se omite el análisis avanzado")
        
        return jsonify({
            'success': True,
            'steps_count': result['steps_count'],
            'message': f'Sesión procesada con {result["steps_count"]} pasos.'
        })
    except Exception as e:
        logger.error(f"Error al procesar sesión: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/generate_documentation/<session_id>', methods=['POST'])
def generate_documentation(session_id):
    """Genera documentación para una sesión procesada."""
    try:
        # Obtener parámetros
        title = request.form.get('title', 'Manual de Usuario')
        author = request.form.get('author', 'Sistema Automático')
        formats = request.form.getlist('formats') or ['markdown', 'html']
        
        # Verificar que la sesión existe
        session_dir = os.path.join(app.config['CAPTURES_DIR'], session_id)
        if not os.path.exists(session_dir):
            return jsonify({
                'success': False,
                'error': f'Sesión no encontrada: {session_id}'
            }), 404
        
        # Verificar que la sesión está procesada
        processed_file = os.path.join(session_dir, "processed_data.json")
        if not os.path.exists(processed_file):
            return jsonify({
                'success': False,
                'error': 'La sesión no ha sido procesada'
            }), 400
        
        # Cargar datos analizados si existen
        analyzed_data = {}
        analysis_file = os.path.join(session_dir, "analyzed_data.json")
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                analyzed_data = json.load(f)
        
        # Generar documentación
        logger.info(f"Generando documentación para sesión: {session_id}")
        logger.info(f"Formatos solicitados: {formats}")
        
        result = doc_generator.generate_documentation(
            session_dir=session_dir,
            analyzed_data=analyzed_data,
            title=title,
            author=author,
            formats=formats
        )
        
        if not result['success']:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Error desconocido al generar documentación')
            }), 500
        
        # Preparar rutas relativas para la respuesta
        outputs = {}
        for format_name, file_path in result['outputs'].items():
            if file_path:  # Algunos formatos pueden fallar (como PDF)
                outputs[format_name] = os.path.basename(file_path)
        
        return jsonify({
            'success': True,
            'outputs': outputs,
            'message': 'Documentación generada exitosamente.'
        })
    except Exception as e:
        logger.error(f"Error al generar documentación: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Descarga un archivo generado."""
    return send_from_directory(app.config['OUTPUT_DIR'], filename, as_attachment=True)

@app.route('/view_image/<session_id>/<filename>')
def view_image(session_id, filename):
    """Muestra una imagen capturada."""
    session_dir = os.path.join(app.config['CAPTURES_DIR'], session_id)
    return send_from_directory(session_dir, filename)

@app.route('/sessions')
def list_sessions():
    """Lista las sesiones disponibles."""
    try:
        sessions = []
        
        # Listar directorios en CAPTURES_DIR
        for item in os.listdir(app.config['CAPTURES_DIR']):
            session_dir = os.path.join(app.config['CAPTURES_DIR'], item)
            
            # Verificar que es un directorio
            if os.path.isdir(session_dir):
                # Verificar si tiene eventos
                events_file = os.path.join(session_dir, 'events.json')
                processed_file = os.path.join(session_dir, 'processed_data.json')
                
                status = 'captured'
                if os.path.exists(processed_file):
                    status = 'processed'
                
                # Contar capturas
                captures_count = len([f for f in os.listdir(session_dir) if f.startswith('capture_') and f.endswith('.png')])
                
                # Extraer fecha y hora del ID de sesión
                date_part = item.split('_')[0] if '_' in item else ''
                time_part = item.split('_')[1] if '_' in item and len(item.split('_')) > 1 else ''
                
                sessions.append({
                    'id': item,
                    'date': date_part,
                    'time': time_part,
                    'status': status,
                    'captures_count': captures_count
                })
        
        # Ordenar por fecha/hora (más reciente primero)
        sessions.sort(key=lambda s: s['id'], reverse=True)
        
        return jsonify({
            'success': True,
            'sessions': sessions
        })
    except Exception as e:
        logger.error(f"Error al listar sesiones: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/session_info/<session_id>')
def session_info(session_id):
    """Obtiene información detallada de una sesión."""
    try:
        # Verificar que la sesión existe
        session_dir = os.path.join(app.config['CAPTURES_DIR'], session_id)
        if not os.path.exists(session_dir):
            return jsonify({
                'success': False,
                'error': f'Sesión no encontrada: {session_id}'
            }), 404
        
        # Verificar archivos
        events_file = os.path.join(session_dir, 'events.json')
        processed_file = os.path.join(session_dir, 'processed_data.json')
        metadata_file = os.path.join(session_dir, 'metadata.json')
        
        # Determinar estado
        status = 'captured'
        steps_count = 0
        
        if os.path.exists(processed_file):
            status = 'processed'
            try:
                with open(processed_file, 'r') as f:
                    data = json.load(f)
                    steps_count = len(data.get('steps', []))
            except:
                pass
        
        # Cargar metadatos si existen
        metadata = {}
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except:
                pass
        
        # Contar capturas
        captures = [f for f in os.listdir(session_dir) if f.startswith('capture_') and f.endswith('.png')]
        
        # Obtener eventos
        events_count = 0
        if os.path.exists(events_file):
            try:
                with open(events_file, 'r') as f:
                    events = json.load(f)
                    events_count = len(events)
            except:
                pass
        
        return jsonify({
            'success': True,
            'session': {
                'id': session_id,
                'status': status,
                'captures_count': len(captures),
                'events_count': events_count,
                'steps_count': steps_count,
                'context': metadata.get('context', ''),
                'captures': [{'filename': f} for f in captures[:10]]  # Limitar a 10 para la vista previa
            }
        })
    except Exception as e:
        logger.error(f"Error al obtener información de sesión: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Sistema de Documentación Automatizada')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host para el servidor web')
    parser.add_argument('--port', type=int, default=5000, help='Puerto para el servidor web')
    parser.add_argument('--debug', action='store_true', help='Ejecutar en modo debug')
    parser.add_argument('--no-multimodal', action='store_true', help='Deshabilitar modelo multimodal')
    parser.add_argument('--model', type=str, help='Modelo multimodal a utilizar')
    parser.add_argument('--quantization', type=str, choices=['4bit', '8bit', 'none'], default='8bit', 
                        help='Nivel de cuantización para el modelo')
    
    return parser.parse_args()

if __name__ == '__main__':
    # Parsear argumentos
    args = parse_args()
    
    # Actualizar configuración según argumentos
    if args.model:
        app.config['LLM_MODEL'] = args.model
    
    if args.quantization == '4bit':
        app.config['LOAD_IN_4BIT'] = True
        app.config['LOAD_IN_8BIT'] = False
    elif args.quantization == '8bit':
        app.config['LOAD_IN_8BIT'] = True
        app.config['LOAD_IN_4BIT'] = False
    elif args.quantization == 'none':
        app.config['LOAD_IN_8BIT'] = False
        app.config['LOAD_IN_4BIT'] = False
    
    # Inicializar componentes
    init_components(args)
    
    # Iniciar servidor
    logger.info(f"Iniciando servidor en {args.host}:{args.port}")
    app.run(debug=args.debug, host=args.host, port=args.port)
