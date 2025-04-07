"""
Script para ejecutar pruebas del sistema y verificar su funcionamiento

Este script ejecuta pruebas básicas para verificar que todos los componentes
del sistema funcionan correctamente.
"""

import os
import sys
import time
import argparse
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("system_test")

def test_screen_capture(output_dir):
    """Prueba el módulo de captura de pantalla."""
    logger.info("Probando módulo de captura de pantalla...")
    
    from src.screen_capture import create_capture_manager
    
    # Crear gestor de captura
    capture_manager = create_capture_manager(output_dir=output_dir)
    
    # Iniciar captura
    logger.info("Iniciando captura de prueba (duración: 3 segundos)...")
    session_id = capture_manager.start_capture(context="Prueba automatizada")
    
    # Esperar 3 segundos
    time.sleep(3)
    
    # Detener captura
    result = capture_manager.stop_capture()
    
    # Verificar resultados
    logger.info(f"Captura finalizada. ID de sesión: {result['session_id']}")
    logger.info(f"Eventos registrados: {result['events_count']}")
    
    return result['session_dir']

def test_image_processing(session_dir):
    """Prueba el módulo de procesamiento de imágenes."""
    logger.info("Probando módulo de procesamiento de imágenes...")
    
    from src.image_processor import create_image_processor
    
    # Crear procesador
    processor = create_image_processor()
    
    # Procesar sesión
    logger.info(f"Procesando sesión en: {session_dir}")
    result = processor.process_session(session_dir)
    
    # Verificar resultados
    if result["success"]:
        logger.info(f"Procesamiento completado. Pasos detectados: {result['steps_count']}")
        logger.info(f"Tipo de flujo de trabajo: {result.get('workflow_type', 'desconocido')}")
    else:
        logger.error(f"Error en procesamiento: {result.get('error', 'Error desconocido')}")
        sys.exit(1)
    
    return result

def test_multimodal_llm(session_dir, use_multimodal=True):
    """Prueba el módulo de análisis multimodal."""
    if not use_multimodal:
        logger.info("Omitiendo prueba de modelo multimodal (--no-multimodal)")
        return {"step_analyses": {}, "workflow_summary": {"summary": "Resumen de prueba"}}
    
    logger.info("Probando módulo de análisis multimodal...")
    
    try:
        from src.llm_multimodal import create_multimodal_llm_engine
        
        # Cargar datos procesados
        import json
        processed_file = os.path.join(session_dir, "processed_data.json")
        
        if not os.path.exists(processed_file):
            logger.error(f"Archivo de datos procesados no encontrado: {processed_file}")
            return {"step_analyses": {}, "workflow_summary": {"summary": "Resumen de prueba"}}
        
        with open(processed_file, 'r') as f:
            processed_data = json.load(f)
        
        # Crear motor multimodal
        logger.info("Inicializando motor multimodal (puede tardar unos minutos)...")
        engine = create_multimodal_llm_engine(load_in_8bit=True)
        
        # Analizar un paso de ejemplo
        if processed_data.get("steps"):
            logger.info("Analizando un paso de ejemplo...")
            step = processed_data["steps"][0]
            step_analysis = engine.analyze_step(step, context=processed_data.get("context", ""))
            
            if step_analysis["success"]:
                logger.info("Análisis de paso completado correctamente")
                logger.info(f"Longitud de descripción generada: {len(step_analysis['description'])} caracteres")
            else:
                logger.warning(f"Error en análisis de paso: {step_analysis.get('description', 'Error desconocido')}")
        
        # Analizar flujo de trabajo
        logger.info("Analizando flujo de trabajo completo...")
        workflow_summary = engine.analyze_workflow(processed_data)
        
        if workflow_summary["success"]:
            logger.info("Análisis de flujo de trabajo completado correctamente")
            logger.info(f"Longitud de resumen generado: {len(workflow_summary['summary'])} caracteres")
        else:
            logger.warning(f"Error en análisis de flujo: {workflow_summary.get('summary', 'Error desconocido')}")
        
        # Descargar modelo para liberar memoria
        engine.unload_model()
        
        # Crear estructura de resultados
        analyzed_data = {
            "step_analyses": {
                "0": step_analysis if processed_data.get("steps") else {}
            },
            "workflow_summary": workflow_summary
        }
        
        # Guardar análisis
        analysis_file = os.path.join(session_dir, "analyzed_data.json")
        with open(analysis_file, 'w') as f:
            json.dump(analyzed_data, f, indent=2)
        
        logger.info(f"Análisis guardado en: {analysis_file}")
        
        return analyzed_data
        
    except Exception as e:
        logger.error(f"Error en prueba de modelo multimodal: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"step_analyses": {}, "workflow_summary": {"summary": "Resumen de prueba"}}

def test_doc_generator(session_dir, analyzed_data):
    """Prueba el módulo de generación de documentación."""
    logger.info("Probando módulo de generación de documentación...")
    
    from src.doc_generator import create_document_generator
    
    # Crear generador
    templates_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "output")
    
    generator = create_document_generator(templates_dir=templates_dir, output_dir=output_dir)
    
    # Generar documentación
    logger.info("Generando documentación de prueba...")
    result = generator.generate_documentation(
        session_dir=session_dir,
        analyzed_data=analyzed_data,
        title="Documentación de Prueba",
        author="Sistema de Pruebas",
        formats=["markdown", "html"]
    )
    
    # Verificar resultados
    if result["success"]:
        logger.info("Documentación generada correctamente")
        for format_name, file_path in result["outputs"].items():
            if file_path:
                logger.info(f"Archivo {format_name.upper()}: {file_path}")
    else:
        logger.error(f"Error en generación de documentación: {result.get('error', 'Error desconocido')}")
    
    return result

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Prueba del Sistema de Documentación Automatizada")
    parser.add_argument("--no-multimodal", action="store_true", help="Omitir prueba de modelo multimodal")
    parser.add_argument("--output-dir", type=str, default="data/test_captures", help="Directorio de salida para capturas")
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=== Iniciando pruebas del sistema ===")
    
    # Probar captura de pantalla
    session_dir = test_screen_capture(args.output_dir)
    
    # Probar procesamiento de imágenes
    test_image_processing(session_dir)
    
    # Probar análisis multimodal
    analyzed_data = test_multimodal_llm(session_dir, not args.no_multimodal)
    
    # Probar generación de documentación
    test_doc_generator(session_dir, analyzed_data)
    
    logger.info("=== Pruebas completadas exitosamente ===")
    logger.info(f"Sesión de prueba: {session_dir}")

if __name__ == "__main__":
    main()
