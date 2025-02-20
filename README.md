---
title: Sugerencia de practica
tags:
  - visionanimacioncomputadora
---

## Sugerencia de Practica. Imágenes Panorámicas

Procesar un conjunto de imágenes para generar una imagen panorámica combinando múltiples vistas de una escena. Utiliza algoritmos de detección y descripción de características (FAST, BRIEF) junto con un emparejamiento de puntos clave (RANSAC) para calcular la transformación necesaria que alinea las imágenes.

## Pasos Previos

1. Descarga el código del siguiente [repositorio](https://github.com/miguehm/descriptor) 
2. Asegúrate de instalar las librerías necesarias:
  ```bash
  pip install opencv-contrib-python matplotlib
  ```
3. Ejecuta el archivo `main.py`

## Explicación del Código

### Paso 1: Carga y Preprocesamiento de Imágenes

- Se carga un conjunto de imágenes desde un array.
- Cada imagen se convierte a escala de grises para facilitar la detección de características.
- **Se aplica un filtro de suavisado Gaussiano para reducir ruido.**

```python
# Mayor tamaño del kernel = Mayor suavisado
gaussian_kernel_size = 15

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray = cv2.GaussianBlur(image_gray, (gaussian_kernel_size, gaussian_kernel_size), 0)
```

### Paso 2: Detección de Características con FAST

- Se utiliza el detector FAST para encontrar puntos clave en la imagen.
- FAST es un detector rápido que identifica esquinas y bordes característicos.

```python
fast = cv2.FastFeatureDetector_create()
keypoints = fast.detect(image_gray, None)
```

### Paso 3: Extracción de Descriptores con BRIEF

- Se aplica el algoritmo BRIEF para describir los puntos clave detectados.
- Los descriptores se utilizan posteriormente para encontrar coincidencias entre imágenes adyacentes.

```python
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
keypoints, descriptors = brief.compute(image_gray, keypoints)
```

### Paso 4: Emparejamiento de Características entre Imágenes

- Se utiliza `BFMatcher` para encontrar coincidencias entre los descriptores de dos imágenes consecutivas.
- Se aplican filtros como `cv2.RANSAC` para eliminar emparejamientos erróneos.

```python
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)
```

### Paso 5: Cálculo de la Homografía

- Se obtienen los puntos de coincidencia en ambas imágenes.
- Se calcula la matriz de homografía usando `cv2.findHomography` con RANSAC.

```python
pts_img1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
pts_img2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
H, _ = cv2.findHomography(pts_img2, pts_img1, cv2.RANSAC, 5.0)
```

### Paso 6: Transformación y Combinación de Imágenes

- Se aplica `warpPerspective` para alinear la segunda imagen con la primera.
- Se fusionan las imágenes transformadas en un solo panorama.

```python
image_warped = cv2.warpPerspective(image2, H, (width, height))
```

## Actividad

1. Regresa al archivo `main.py` y dirígete a la linea 43

```py
gaussian_kernel_size = 15
```

2. Aumenta el valor de `gaussian_kernel_size`, por ejemplo, de diez en diez y vuelve a ejecutar.

3. Continua repitiendo el paso dos, presta atención a los mensajes de la consola.

- ¿Que sucedió cuando el valor asignado a `gaussian_kernel_size` fue muy grande?
- ¿Cuál consideras la razón principal de ese comportamiento y su relación con el descriptor de características?

Fuente de consulta sugerida: [Feature descriptors](https://medium.com/@deepanshut041/introduction-to-brief-binary-robust-independent-elementary-features-436f4a31a0e6)
