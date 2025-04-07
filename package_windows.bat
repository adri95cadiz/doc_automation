@echo off
echo Empaquetando Sistema de Documentacion Automatizada para Windows...

:: Verificar si Python está instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python no esta instalado. Por favor, instale Python 3.8 o superior.
    pause
    exit /b 1
)

:: Crear directorio temporal para empaquetado
set TEMP_DIR=temp_package
if exist %TEMP_DIR% rmdir /s /q %TEMP_DIR%
mkdir %TEMP_DIR%

:: Copiar archivos necesarios
echo Copiando archivos...
mkdir %TEMP_DIR%\src
mkdir %TEMP_DIR%\templates
mkdir %TEMP_DIR%\data
mkdir %TEMP_DIR%\data\captures
mkdir %TEMP_DIR%\data\output
mkdir %TEMP_DIR%\data\models
mkdir %TEMP_DIR%\tests

:: Copiar archivos de código fuente
copy src\*.py %TEMP_DIR%\src\
copy app.py %TEMP_DIR%\
copy requirements.txt %TEMP_DIR%\
copy windows_install.bat %TEMP_DIR%\install.bat
copy README.md %TEMP_DIR%\

:: Copiar plantillas
copy templates\*.* %TEMP_DIR%\templates\

:: Copiar pruebas
copy tests\*.py %TEMP_DIR%\tests\

:: Crear archivo ZIP
echo Creando archivo ZIP...
powershell Compress-Archive -Path %TEMP_DIR%\* -DestinationPath doc_automation_v3_windows.zip -Force

:: Limpiar
echo Limpiando archivos temporales...
rmdir /s /q %TEMP_DIR%

echo Empaquetado completado: doc_automation_v3_windows.zip
pause
