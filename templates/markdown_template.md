
# {{ title }}

**Autor:** {{ author }}  
**Fecha:** {{ date }}

## Introduccion

{{ summary }}

{% for step in steps %}
## {{ step.title }}

![Captura de pantalla]({{ step.image_path }})

{{ step.description }}

{% endfor %}