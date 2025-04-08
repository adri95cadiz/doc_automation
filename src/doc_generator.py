"""
Módulo de generación de documentación adaptado para Windows

Este módulo se encarga de generar documentación estructurada a partir de los análisis
de capturas de pantalla y descripciones generadas por el modelo multimodal.
"""

import os
import json
import logging
import datetime
import jinja2
import markdown
import shutil
from pathlib import Path

# Importar utilidades de plataforma
from src.platform_utils import normalize_path, ensure_dir, get_platform

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("doc_generator.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("doc_generator")

class DocumentGenerator:
    """
    Generador de documentación a partir de datos procesados y análisis multimodal.
    """
    
    def __init__(self, llm_processor, image_processor=None, video_processor=None):
        """
        Inicializa el generador de documentación.
        
        Args:
            llm_processor: Procesador de lenguaje natural
            image_processor: Procesador de imágenes (opcional)
            video_processor: Procesador de video (opcional)
        """
        self.llm_processor = llm_processor
        self.image_processor = image_processor
        self.video_processor = video_processor
        
    def process_session(self, session_data):
        """
        Procesa una sesión de captura y genera la documentación.
        
        Args:
            session_data: Datos de la sesión (puede incluir imágenes o video)
            
        Returns:
            dict: Datos procesados para generar documentación
        """
        processed_data = {
            'workflow_summary': '',
            'steps': []
        }
        
        try:
            # Procesar video si está disponible
            if 'video_path' in session_data and self.video_processor:
                keyframe_dir = os.path.join(session_data['output_dir'], 'keyframes')
                keyframes = self.video_processor.extract_keyframes(
                    session_data['video_path'], 
                    keyframe_dir
                )
                
                if keyframes:
                    processed_data['steps'] = self._process_keyframes(keyframes)
                    
            # Procesar imágenes si no hay video o si falló el procesamiento de video
            elif 'captures' in session_data and self.image_processor:
                processed_data['steps'] = self._process_captures(session_data['captures'])
                
            # Generar resumen del workflow
            if processed_data['steps']:
                processed_data['workflow_summary'] = self.llm_processor.generate_workflow_summary(
                    processed_data['steps']
                )
                
            return processed_data
            
        except Exception as e:
            logger.error(f"Error al procesar sesión: {str(e)}")
            return processed_data
            
    def _process_keyframes(self, keyframe_paths):
        """
        Procesa los frames clave extraídos del video.
        
        Args:
            keyframe_paths: Lista de rutas a los frames
            
        Returns:
            list: Lista de pasos procesados
        """
        steps = []
        
        for frame_path in keyframe_paths:
            try:
                # Extraer texto de la imagen
                extracted_text = self.image_processor.extract_text(frame_path)
                
                # Generar descripción usando el LLM
                description = self.llm_processor.generate_description(
                    frame_path,
                    extracted_text
                )
                
                steps.append({
                    'image_path': frame_path,
                    'text': extracted_text,
                    'description': description
                })
                
            except Exception as e:
                logger.error(f"Error al procesar frame {frame_path}: {str(e)}")
                continue
                
        return steps
        
    def _process_captures(self, captures):
        """
        Procesa las capturas de pantalla individuales.
        
        Args:
            captures: Lista de rutas a las capturas
            
        Returns:
            list: Lista de pasos procesados
        """
        steps = []
        
        for capture in captures:
            try:
                # Extraer texto de la imagen
                extracted_text = self.image_processor.extract_text(capture)
                
                # Generar descripción usando el LLM
                description = self.llm_processor.generate_description(
                    capture,
                    extracted_text
                )
                
                steps.append({
                    'image_path': capture,
                    'text': extracted_text,
                    'description': description
                })
                
            except Exception as e:
                logger.error(f"Error al procesar captura {capture}: {str(e)}")
                continue
                
        return steps
    
    def generate_documentation(self, session_dir, analyzed_data, title="Manual de Usuario", 
                              author="Sistema Automático", formats=None):
        """
        Genera documentación a partir de datos analizados.
        
        Args:
            session_dir: Directorio de la sesión
            analyzed_data: Datos analizados por el modelo multimodal
            title: Título del documento
            author: Autor del documento
            formats: Lista de formatos a generar (markdown, html, pdf)
            
        Returns:
            dict: Resultados de la generación
        """
        try:
            # Normalizar ruta
            session_dir = normalize_path(session_dir)
            
            # Configurar formatos por defecto
            if formats is None:
                formats = ["markdown", "html"]
            
            logger.info(f"Generando documentación para sesión en: {session_dir}")
            logger.info(f"Formatos solicitados: {formats}")
            
            # Cargar datos procesados
            processed_file = os.path.join(session_dir, "processed_data.json")
            if not os.path.exists(processed_file):
                logger.error(f"Archivo de datos procesados no encontrado: {processed_file}")
                return {"success": False, "error": "Datos procesados no encontrados"}
            
            with open(processed_file, 'r') as f:
                processed_data = json.load(f)
            
            # Preparar directorio de salida específico para esta documentación
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            doc_id = f"{title.lower().replace(' ', '_')}_{timestamp}"
            # Eliminar caracteres no válidos para nombres de archivo en Windows
            doc_id = ''.join(c for c in doc_id if c.isalnum() or c in ['_', '-'])
            
            doc_dir = os.path.join(self.output_dir, doc_id)
            ensure_dir(doc_dir)
            
            # Crear directorio para imágenes
            images_dir = os.path.join(doc_dir, "images")
            ensure_dir(images_dir)
            
            # Preparar datos para plantillas
            template_data = self._prepare_template_data(
                processed_data, 
                analyzed_data, 
                title, 
                author, 
                session_dir, 
                images_dir
            )
            
            # Generar documentación en los formatos solicitados
            outputs = {}
            
            if "markdown" in formats:
                md_file = self._generate_markdown(template_data, doc_dir)
                outputs["markdown"] = md_file
            
            if "html" in formats:
                html_file = self._generate_html(template_data, doc_dir)
                outputs["html"] = html_file
            
            if "pdf" in formats:
                pdf_file = self._generate_pdf(template_data, doc_dir)
                outputs["pdf"] = pdf_file
            
            logger.info(f"Documentación generada exitosamente en: {doc_dir}")
            
            return {
                "success": True,
                "doc_id": doc_id,
                "doc_dir": doc_dir,
                "outputs": outputs
            }
            
        except Exception as e:
            logger.error(f"Error al generar documentación: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    def _prepare_template_data(self, processed_data, analyzed_data, title, author, session_dir, images_dir):
        """
        Prepara los datos para las plantillas.
        
        Args:
            processed_data: Datos procesados de la sesión
            analyzed_data: Datos analizados por el modelo multimodal
            title: Título del documento
            author: Autor del documento
            session_dir: Directorio de la sesión
            images_dir: Directorio para imágenes en la documentación
            
        Returns:
            dict: Datos preparados para plantillas
        """
        # Extraer información relevante
        context = processed_data.get("context", "Sin contexto")
        workflow_type = processed_data.get("workflow_type", "general")
        steps = processed_data.get("steps", [])
        
        # Obtener resumen del flujo de trabajo
        workflow_summary = analyzed_data.get("workflow_summary", {}).get("summary", "")
        
        # Preparar pasos con imágenes copiadas
        prepared_steps = []
        
        for i, step in enumerate(steps):
            # Obtener análisis del paso
            step_analysis = analyzed_data.get("step_analyses", {}).get(str(i), {})
            
            # Obtener ruta de imagen original
            original_image_path = step.get("image", {}).get("path", "")
            
            # Si no hay imagen, continuar con el siguiente paso
            if not original_image_path or not os.path.exists(original_image_path):
                logger.warning(f"Imagen no encontrada para paso {i+1}: {original_image_path}")
                continue
            
            # Copiar imagen al directorio de documentación
            image_filename = f"step_{i+1}_{os.path.basename(original_image_path)}"
            # Asegurar que el nombre de archivo sea válido en Windows
            image_filename = ''.join(c for c in image_filename if c.isalnum() or c in ['_', '-', '.'])
            
            image_path = os.path.join(images_dir, image_filename)
            shutil.copy2(original_image_path, image_path)
            
            # Ruta relativa para la plantilla (usar siempre / para HTML)
            relative_image_path = f"images/{image_filename}"
            
            # Preparar datos del paso
            prepared_step = {
                "number": i + 1,
                "title": f"Paso {i + 1}",
                "description": step_analysis.get("description", step.get("description", "")),
                "image_path": relative_image_path,
                "events": step.get("events", []),
                "pattern": step.get("pattern", "")
            }
            
            prepared_steps.append(prepared_step)
        
        # Datos completos para plantilla
        template_data = {
            "title": title,
            "author": author,
            "date": datetime.datetime.now().strftime("%d/%m/%Y"),
            "context": context,
            "workflow_type": workflow_type,
            "summary": workflow_summary,
            "steps": prepared_steps,
            "metadata": {
                "generated_at": datetime.datetime.now().isoformat(),
                "session_id": os.path.basename(session_dir),
                "steps_count": len(prepared_steps),
                "platform": self.platform
            }
        }
        
        return template_data
    
    def _generate_markdown(self, template_data, doc_dir):
        """
        Genera documentación en formato Markdown.
        
        Args:
            template_data: Datos para la plantilla
            doc_dir: Directorio de documentación
            
        Returns:
            str: Ruta al archivo generado
        """
        try:
            # Cargar plantilla
            template = self.jinja_env.get_template("markdown_template.md")
            
            # Renderizar plantilla
            content = template.render(**template_data)
            
            # Guardar archivo
            safe_title = ''.join(c for c in template_data['title'].replace(' ', '_') if c.isalnum() or c in ['_', '-'])
            output_file = os.path.join(doc_dir, f"{safe_title}.md")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Documentación Markdown generada: {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error al generar Markdown: {str(e)}")
            raise
    
    def _generate_html(self, template_data, doc_dir):
        """
        Genera documentación en formato HTML.
        
        Args:
            template_data: Datos para la plantilla
            doc_dir: Directorio de documentación
            
        Returns:
            str: Ruta al archivo generado
        """
        try:
            # Cargar plantilla
            template = self.jinja_env.get_template("html_template.html")
            
            # Renderizar plantilla
            content = template.render(**template_data)
            
            # Guardar archivo
            safe_title = ''.join(c for c in template_data['title'].replace(' ', '_') if c.isalnum() or c in ['_', '-'])
            output_file = os.path.join(doc_dir, f"{safe_title}.html")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Documentación HTML generada: {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error al generar HTML: {str(e)}")
            
            # Intentar generar HTML a partir del Markdown como fallback
            try:
                md_file = self._generate_markdown(template_data, doc_dir)
                
                with open(md_file, 'r', encoding='utf-8') as f:
                    md_content = f.read()
                
                html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
                
                # Añadir estilos básicos
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <title>{template_data['title']}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; }}
                        img {{ max-width: 100%; }}
                        h1 {{ color: #2c3e50; }}
                        h2 {{ color: #3498db; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
                        .step {{ margin-bottom: 30px; }}
                    </style>
                </head>
                <body>
                    {html_content}
                </body>
                </html>
                """
                
                # Guardar archivo
                safe_title = ''.join(c for c in template_data['title'].replace(' ', '_') if c.isalnum() or c in ['_', '-'])
                output_file = os.path.join(doc_dir, f"{safe_title}.html")
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                logger.info(f"Documentación HTML generada (fallback): {output_file}")
                
                return output_file
                
            except Exception as inner_e:
                logger.error(f"Error en fallback HTML: {str(inner_e)}")
                raise e
    
    def _generate_pdf(self, template_data, doc_dir):
        """
        Genera documentación en formato PDF.
        
        Args:
            template_data: Datos para la plantilla
            doc_dir: Directorio de documentación
            
        Returns:
            str: Ruta al archivo generado
        """
        try:
            # Primero generar HTML
            html_file = self._generate_html(template_data, doc_dir)
            
            # Intentar convertir a PDF usando weasyprint
            try:
                # En Windows, verificar si weasyprint está disponible
                if self.platform == 'windows':
                    logger.info("Verificando disponibilidad de WeasyPrint en Windows...")
                
                import weasyprint
                
                # Ruta de salida
                safe_title = ''.join(c for c in template_data['title'].replace(' ', '_') if c.isalnum() or c in ['_', '-'])
                output_file = os.path.join(doc_dir, f"{safe_title}.pdf")
                
                # Convertir HTML a PDF
                html = weasyprint.HTML(filename=html_file)
                html.write_pdf(output_file)
                
                logger.info(f"Documentación PDF generada: {output_file}")
                
                return output_file
                
            except ImportError:
                logger.warning("WeasyPrint no está instalado, intentando con wkhtmltopdf...")
                
                # Intentar con wkhtmltopdf
                try:
                    import subprocess
                    
                    # Ruta de salida
                    safe_title = ''.join(c for c in template_data['title'].replace(' ', '_') if c.isalnum() or c in ['_', '-'])
                    output_file = os.path.join(doc_dir, f"{safe_title}.pdf")
                    
                    # Comando para wkhtmltopdf (ajustar según plataforma)
                    if self.platform == 'windows':
                        # En Windows, buscar en ubicaciones comunes
                        wkhtmltopdf_cmd = "wkhtmltopdf.exe"
                        common_paths = [
                            r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe",
                            r"C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe"
                        ]
                        
                        for path in common_paths:
                            if os.path.exists(path):
                                wkhtmltopdf_cmd = path
                                break
                    else:
                        wkhtmltopdf_cmd = "wkhtmltopdf"
                    
                    # Ejecutar wkhtmltopdf
                    subprocess.run([wkhtmltopdf_cmd, html_file, output_file], check=True)
                    
                    logger.info(f"Documentación PDF generada (wkhtmltopdf): {output_file}")
                    
                    return output_file
                    
                except (ImportError, subprocess.SubprocessError) as e:
                    logger.error(f"Error al generar PDF con wkhtmltopdf: {str(e)}")
                    logger.warning("No se pudo generar PDF, se proporcionará solo HTML")
                    return None
                
        except Exception as e:
            logger.error(f"Error al generar PDF: {str(e)}")
            return None

def create_document_generator(templates_dir="templates", output_dir="data/output"):
    """
    Crea una instancia del generador de documentación.
    
    Args:
        templates_dir: Directorio de plantillas
        output_dir: Directorio de salida para documentación generada
        
    Returns:
        DocumentGenerator: Instancia del generador
    """
    return DocumentGenerator(templates_dir, output_dir)

# Función para pruebas independientes
if __name__ == "__main__":
    import sys
    import argparse
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="Generador de documentación")
    parser.add_argument("--session", type=str, required=True, help="Directorio de la sesión")
    parser.add_argument("--analyzed", type=str, required=True, help="Archivo JSON con datos analizados")
    parser.add_argument("--title", type=str, default="Manual de Usuario", help="Título del documento")
    parser.add_argument("--author", type=str, default="Sistema Automático", help="Autor del documento")
    parser.add_argument("--formats", type=str, default="markdown,html", help="Formatos a generar (separados por comas)")
    
    args = parser.parse_args()
    
    # Cargar datos analizados
    with open(args.analyzed, 'r') as f:
        analyzed_data = json.load(f)
    
    # Crear generador
    generator = create_document_generator()
    
    # Generar documentación
    formats = args.formats.split(',')
    result = generator.generate_documentation(
        session_dir=args.session,
        analyzed_data=analyzed_data,
        title=args.title,
        author=args.author,
        formats=formats
    )
    
    # Mostrar resultados
    if result["success"]:
        print(f"Documentación generada exitosamente en: {result['doc_dir']}")
        for fmt, path in result["outputs"].items():
            if path:
                print(f"- {fmt.upper()}: {path}")
    else:
        print(f"Error: {result['error']}")
        sys.exit(1)
