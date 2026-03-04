# Detector de objetos (MobileNetSSD + OpenCV DNN)

## Descripción (≤350 caracteres)
Detector de objetos en tiempo real con webcam usando OpenCV (DNN) y MobileNetSSD (Caffe). Carga `MobileNetSSD_deploy.prototxt` y `MobileNetSSD_deploy.caffemodel`, ejecuta inferencia por frame y dibuja cajas + etiqueta si `confidence > 0.75`.

## Archivos del proyecto
- `deep.py`: script principal de inferencia por cámara.
- `MobileNetSSD_deploy.prototxt`: definición de la red (Caffe).
- `MobileNetSSD_deploy.caffemodel`: pesos entrenados (Caffe).

## Requisitos
- Python 3.x
- OpenCV con soporte DNN (`opencv-python` u `opencv-contrib-python`)

Instalación:
    pip install opencv-python

## Ejecución
1. Colocá estos archivos en la misma carpeta:
   - `deep.py`
   - `MobileNetSSD_deploy.prototxt`
   - `MobileNetSSD_deploy.caffemodel`

2. Ejecutá:
    python deep.py

Se abrirá una ventana llamada **`deteccion`** con el video de la cámara y las detecciones.

## Controles
- Presioná **`q`** para salir.

## Cómo funciona (resumen técnico)
- Abre la cámara: `cv2.VideoCapture(0, cv2.CAP_DSHOW)`.
- Convierte cada frame a *blob* (300×300) con:
  - `scalefactor = 0.007843`
  - `size = (300, 300)`
  - `mean = (127.5, 127.5, 127.5)`
- Ejecuta inferencia con `net.forward()`.
- Filtra detecciones por `confidence > 0.75`.
- Dibuja rectángulo y etiqueta con `cv2.rectangle` y `cv2.putText`.

## Clases incluidas
El script define un mapeo de IDs a etiquetas (en español), incluyendo, entre otras: Avion, Bicicleta, Ave, Bote, Botella, Colectivo, Auto, Gato, Silla, Vaca, Mesa, Perro, Caballo, Moto, Persona, etc.
