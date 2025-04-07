#!/bin/bash

# Script de instalación para el Sistema de Documentación Automatizada
# Este script instala todas las dependencias necesarias para ejecutar el sistema

echo "=== Instalando Sistema de Documentación Automatizada ==="
echo "Creando entorno virtual..."

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

echo "Instalando dependencias básicas..."

# Instalar dependencias básicas
pip install --upgrade pip
pip install flask pillow pyautogui pynput opencv-python pytesseract jinja2 markdown

echo "Instalando dependencias para el modelo multimodal..."

# Instalar dependencias para el modelo multimodal
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate bitsandbytes

echo "Instalando dependencias para generación de PDF..."

# Instalar dependencias para generación de PDF
pip install weasyprint

echo "Instalando dependencias para desarrollo y pruebas..."

# Instalar dependencias para desarrollo y pruebas
pip install pytest pytest-flask

echo "Creando directorios necesarios..."

# Crear directorios necesarios
mkdir -p data/captures data/output data/models

echo "Creando archivos de plantilla..."

# Crear plantillas básicas si no existen
if [ ! -f templates/markdown_template.md ]; then
    echo "Creando plantilla Markdown..."
    cat > templates/markdown_template.md << 'EOL'
# {{ title }}

**Autor:** {{ author }}  
**Fecha:** {{ date }}

## Introducción

{{ summary }}

{% for step in steps %}
## {{ step.title }}

![Captura de pantalla]({{ step.image_path }})

{{ step.description }}

{% endfor %}
EOL
fi

if [ ! -f templates/html_template.html ]; then
    echo "Creando plantilla HTML..."
    cat > templates/html_template.html << 'EOL'
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        h2 {
            color: #3498db;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }
        .step {
            margin-bottom: 30px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }
        .metadata {
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    
    <div class="metadata">
        <p><strong>Autor:</strong> {{ author }}<br>
        <strong>Fecha:</strong> {{ date }}</p>
    </div>
    
    <h2>Introducción</h2>
    <p>{{ summary }}</p>
    
    {% for step in steps %}
    <div class="step">
        <h2>{{ step.title }}</h2>
        <img src="{{ step.image_path }}" alt="Captura de pantalla del paso {{ step.number }}">
        <p>{{ step.description }}</p>
    </div>
    {% endfor %}
    
    <footer>
        <p><small>Generado automáticamente por el Sistema de Documentación Automatizada</small></p>
    </footer>
</body>
</html>
EOL
fi

echo "Instalación completada."
echo "Para ejecutar el sistema: source venv/bin/activate && python app.py"
