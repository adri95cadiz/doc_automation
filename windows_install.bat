@echo off
echo === Instalando Sistema de Documentacion Automatizada para Windows ===
echo Creando entorno virtual...

:: Verificar si Python está instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python no esta instalado. Por favor, instale Python 3.8 o superior.
    echo Puede descargarlo desde https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Crear entorno virtual
python -m venv venv
call venv\Scripts\activate.bat

echo Instalando dependencias basicas...

:: Instalar dependencias básicas
python -m pip install --upgrade pip
python -m pip install flask pillow pyautogui pynput opencv-python jinja2 markdown

echo Instalando dependencias para el modelo multimodal...

:: Instalar dependencias para el modelo multimodal
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install transformers accelerate bitsandbytes

echo Instalando dependencias para desarrollo y pruebas...

:: Instalar dependencias para desarrollo y pruebas
python -m pip install pytest pytest-flask

python -m pip install -r requirements.txt

echo Creando directorios necesarios...

:: Crear directorios necesarios
mkdir data\captures data\output data\models 2>nul

echo Creando archivos de plantilla...

:: Crear plantillas básicas si no existen
if not exist templates\markdown_template.md (
    echo Creando plantilla Markdown...
)
if not exist templates\markdown_template.md (
    (
        echo # {{ title }}
        echo.
        echo **Autor:** {{ author }}  
        echo **Fecha:** {{ date }}
        echo.
        echo ## Introduccion
        echo.
        echo {{ summary }}
        echo.
        echo {% for step in steps %}
        echo ## {{ step.title }}
        echo.
        echo ![Captura de pantalla]({{ step.image_path }})
        echo.
        echo {{ step.description }}
        echo.
        echo {% endfor %}
    ) > templates\markdown_template.md
)

if not exist templates\html_template.html (
    echo Creando plantilla HTML...
)
if not exist templates\html_template.html (
    (
        echo ^<!DOCTYPE html^>
        echo ^<html lang="es"^>
        echo ^<head^>
        echo     ^<meta charset="UTF-8"^>
        echo     ^<meta name="viewport" content="width=device-width, initial-scale=1.0"^>
        echo     ^<title^>{{ title }}^</title^>
        echo     ^<style^>
        echo         body {
        echo             font-family: Arial, sans-serif;
        echo             line-height: 1.6;
        echo             max-width: 900px;
        echo             margin: 0 auto;
        echo             padding: 20px;
        echo         }
        echo         img {
        echo             max-width: 100%%;
        echo             border: 1px solid #ddd;
        echo             border-radius: 4px;
        echo             padding: 5px;
        echo         }
        echo         h1 {
        echo             color: #2c3e50;
        echo             border-bottom: 2px solid #eee;
        echo             padding-bottom: 10px;
        echo         }
        echo         h2 {
        echo             color: #3498db;
        echo             border-bottom: 1px solid #eee;
        echo             padding-bottom: 5px;
        echo         }
        echo         .step {
        echo             margin-bottom: 30px;
        echo             padding: 15px;
        echo             background-color: #f9f9f9;
        echo             border-radius: 5px;
        echo         }
        echo         .metadata {
        echo             color: #7f8c8d;
        echo             font-size: 0.9em;
        echo             margin-bottom: 20px;
        echo         }
        echo     ^</style^>
        echo ^</head^>
        echo ^<body^>
        echo     ^<h1^>{{ title }}^</h1^>
        echo     
        echo     ^<div class="metadata"^>
        echo         ^<p^>^<strong^>Autor:^</strong^> {{ author }}^<br^>
        echo         ^<strong^>Fecha:^</strong^> {{ date }}^</p^>
        echo     ^</div^>
        echo     
        echo     ^<h2^>Introduccion^</h2^>
        echo     ^<p^>{{ summary }}^</p^>
        echo     
        echo     {% for step in steps %}
        echo     ^<div class="step"^>
        echo         ^<h2^>{{ step.title }}^</h2^>
        echo         ^<img src="{{ step.image_path }}" alt="Captura de pantalla del paso {{ step.number }}"^>
        echo         ^<p^>{{ step.description }}^</p^>
        echo     ^</div^>
        echo     {% endfor %}
        echo     
        echo     ^<footer^>
        echo         ^<p^>^<small^>Generado automaticamente por el Sistema de Documentacion Automatizada^</small^>^</p^>
        echo     ^</footer^>
        echo ^</body^>
        echo ^</html^>
    ) > templates\html_template.html
)

echo Instalacion completada.
echo Para ejecutar el sistema: call venv\Scripts\activate ^&^& python app.py
pause
