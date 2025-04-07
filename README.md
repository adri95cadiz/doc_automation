# Sistema de Documentación Automatizada

Este sistema captura las interacciones del usuario con aplicaciones web, procesa las capturas utilizando modelos multimodales de código abierto, y genera documentación técnica en lenguaje natural de forma automática.

## Características principales

- **Captura inteligente**: Registra automáticamente las acciones del usuario y captura pantallas en momentos clave
- **Análisis multimodal**: Utiliza modelos de IA multimodales para "ver" y comprender lo que sucede en las capturas
- **Generación contextual**: Crea explicaciones detalladas basadas en el contexto proporcionado
- **Ejecución local**: Funciona completamente en local, sin dependencias de servicios en la nube
- **Interfaz web**: Proporciona una interfaz intuitiva para controlar todo el proceso
- **Múltiples formatos**: Genera documentación en Markdown, HTML y PDF

## Requisitos del sistema

- Python 3.8 o superior
- Mínimo 8GB de RAM (16GB recomendado para modelos multimodales)
- GPU compatible con CUDA (opcional, mejora significativamente el rendimiento)
- Tesseract OCR instalado en el sistema

## Instalación

1. Clone este repositorio:
   ```bash
   git clone https://github.com/usuario/doc_automation.git
   cd doc_automation
   ```

2. Ejecute el script de instalación:
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

3. Active el entorno virtual:
   ```bash
   source venv/bin/activate
   ```

## Uso

### Iniciar la aplicación

```bash
python app.py
```

Por defecto, la aplicación estará disponible en http://localhost:5000

### Opciones de línea de comandos

```bash
python app.py --help
```

Opciones disponibles:
- `--host`: Host para el servidor web (por defecto: 0.0.0.0)
- `--port`: Puerto para el servidor web (por defecto: 5000)
- `--debug`: Ejecutar en modo debug
- `--no-multimodal`: Deshabilitar modelo multimodal
- `--model`: Modelo multimodal a utilizar
- `--quantization`: Nivel de cuantización para el modelo (4bit, 8bit, none)

### Flujo de trabajo básico

1. **Iniciar captura**: Configure el título, contexto e intervalo de captura, y haga clic en "Iniciar Captura"
2. **Realizar acciones**: Interactúe normalmente con su aplicación mientras el sistema registra sus acciones
3. **Detener captura**: Cuando termine, haga clic en "Detener Captura"
4. **Procesar sesión**: Seleccione la sesión y haga clic en "Procesar Sesión"
5. **Generar documentación**: Configure el título, autor y formatos deseados, y haga clic en "Crear Documentación"
6. **Descargar resultados**: Descargue la documentación generada en los formatos seleccionados

## Estructura del proyecto

```
doc_automation_v3/
├── app.py                 # Aplicación principal
├── install.sh             # Script de instalación
├── data/                  # Datos generados
│   ├── captures/          # Capturas de pantalla
│   ├── models/            # Modelos descargados
│   └── output/            # Documentación generada
├── src/                   # Código fuente
│   ├── screen_capture.py  # Módulo de captura
│   ├── image_processor.py # Módulo de procesamiento
│   ├── llm_multimodal.py  # Módulo de análisis multimodal
│   └── doc_generator.py   # Módulo de generación
├── templates/             # Plantillas
│   ├── index.html         # Interfaz web
│   ├── markdown_template.md # Plantilla Markdown
│   └── html_template.html # Plantilla HTML
└── tests/                 # Pruebas
    ├── test_system.py     # Pruebas unitarias
    └── run_system_test.py # Prueba del sistema completo
```

## Modelos multimodales compatibles

El sistema está diseñado para funcionar con varios modelos multimodales de código abierto:

- **LLaVA** (por defecto): `llava-hf/llava-1.5-7b-hf`
- **CogVLM**: `THUDM/cogvlm-chat-hf`
- **BLIP-2**: `Salesforce/blip2-opt-2.7b`

Para cambiar el modelo:

```bash
python app.py --model "THUDM/cogvlm-chat-hf"
```

## Optimización de rendimiento

Para sistemas con recursos limitados:

```bash
# Usar cuantización de 4 bits (menor uso de memoria)
python app.py --quantization 4bit

# Deshabilitar completamente el modelo multimodal
python app.py --no-multimodal
```

## Pruebas

Para ejecutar las pruebas unitarias:

```bash
pytest tests/test_system.py
```

Para ejecutar una prueba completa del sistema:

```bash
python tests/run_system_test.py
```

## Solución de problemas

### Error al cargar el modelo multimodal

Si encuentra errores relacionados con la memoria al cargar el modelo multimodal:

1. Intente usar cuantización de 4 bits: `--quantization 4bit`
2. Si persiste el error, ejecute sin capacidades multimodales: `--no-multimodal`

### Problemas con OCR

Si el reconocimiento de texto no funciona correctamente:

1. Verifique que Tesseract OCR esté instalado correctamente
2. Pruebe con otro idioma: `--ocr-lang eng` (para inglés)

## Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abra un issue para discutir los cambios propuestos.
