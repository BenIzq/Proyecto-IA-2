# Traductor de Lenguaje de Señas ASL - Aplicación Web

Esta aplicación web permite traducir lenguaje de señas americano (ASL) a texto a través de la cámara web en tiempo real. Utiliza Flask como framework web, OpenCV para procesamiento de video, MediaPipe para detección de manos y un modelo de aprendizaje profundo previamente entrenado para la clasificación de señas.

## Requisitos previos

- Python 3.7 o superior
- Una cámara web conectada a tu computadora
- Modelo entrenado (`mejor_modelo_asl.pth`)
- Archivo `entrenar_modelo.py` que contiene la clase `ASLTranslator`

## Instalación

1. Clona o descarga este repositorio

2. Instala las dependencias necesarias:

```bash
pip install flask opencv-python mediapipe torch numpy
```

3. Asegúrate de tener los siguientes archivos en el directorio del proyecto:
   - `mejor_modelo_asl.pth` (tu modelo entrenado)
   - `entrenar_modelo.py` (contiene la clase ASLTranslator)

4. Crea la estructura de carpetas necesaria:

```
project_structure/
├── app.py                   # Archivo principal de Flask
├── mejor_modelo_asl.pth     # Tu modelo entrenado
├── entrenar_modelo.py       # Archivo con la clase ASLTranslator
├── templates/               # Carpeta para plantillas HTML
│   └── index.html           # Plantilla de la interfaz web
└── traducciones/            # Carpeta donde se guardarán las traducciones (se creará automáticamente)
```

## Ejecución

1. Ejecuta la aplicación Flask:

```bash
python app.py
```

2. Abre tu navegador y ve a la dirección:

```
http://127.0.0.1:5000/
```

## Cómo usar la aplicación

1. Una vez que la aplicación esté en funcionamiento, verás la transmisión en vivo de tu cámara web.

2. Realiza señas de ASL frente a la cámara:
   - Letras del alfabeto (A-Z)
   - "space" para añadir un espacio
   - "delete" para borrar la última letra

3. Mantén una seña estable durante unos segundos para que sea reconocida.

4. El texto traducido aparecerá debajo del video.

5. Puedes:
   - Borrar el texto usando el botón "Borrar texto"
   - Guardar la traducción actual usando el botón "Guardar traducción"

## Notas

- La aplicación guarda las traducciones en la carpeta `traducciones` con un nombre de archivo que incluye la fecha y hora.
- El modelo debe ser estable para una detección confiable; mantén la mano relativamente quieta cuando realices señas.
- La iluminación adecuada mejora significativamente la precisión del reconocimiento.

## Solución de problemas

- Si la cámara no se inicia, verifica que esté conectada correctamente y que no esté siendo utilizada por otra aplicación.
- Si el modelo no carga, asegúrate de que los archivos `mejor_modelo_asl.pth` y `entrenar_modelo.py` estén presentes en el directorio principal.
- Si la detección no es precisa, intenta mejorar la iluminación o ajustar la posición de la cámara.
