import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ====================================================
# Paso 1: Cargar y preprocesar un array de imágenes
# ====================================================

# Obtener todas las imágenes JPG de la carpeta 'images'
image_dir = 'images'  # Carpeta donde están las imágenes
image_paths = sorted([
    os.path.join(image_dir, f) 
    for f in os.listdir(image_dir) 
    if f.lower().endswith('.jpg')  # Acepta .jpg y .JPG
])

# Inicializar la lista donde se almacenarán las imágenes originales
images = []

for path in image_paths:
    image = cv2.imread(path)
    if image is None:
        print(f"Error al cargar la imagen: {path}")
        continue
    images.append(image)

# Verificar que se haya cargado al menos una imagen
if len(images) == 0:
    print("No se pudo cargar ninguna imagen. Verifica las rutas.")
    exit()

# ====================================================
# Función para unir dos imágenes mediante FAST, BRIEF y homografía
# ====================================================

def stitch_images(img1, img2):
    """
    Une dos imágenes utilizando detección de características (FAST), extracción de descriptores (BRIEF),
    emparejamiento, cálculo de homografía y warpPerspective.
    """
    
    gaussian_kernel_size = 15
    
    # Convertir ambas imágenes a escala de grises y aplicar desenfoque Gaussiano
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (gaussian_kernel_size, gaussian_kernel_size), 0)
    gray2 = cv2.GaussianBlur(gray2, (gaussian_kernel_size, gaussian_kernel_size), 0)
    
    # -----------------------------------------------------
    # Detección de características con FAST y extracción de descriptores con BRIEF
    # -----------------------------------------------------
    fast = cv2.FastFeatureDetector_create()
    keypoints1 = fast.detect(gray1, None)
    keypoints2 = fast.detect(gray2, None)
    
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    keypoints1, descriptors1 = brief.compute(gray1, keypoints1)
    keypoints2, descriptors2 = brief.compute(gray2, keypoints2)
    
    # -----------------------------------------------------
    # Emparejar descriptores entre las dos imágenes
    # -----------------------------------------------------
    # Para descriptores binarios se utiliza NORM_HAMMING
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    
    # Ordenar las coincidencias según la distancia (menor distancia = mejor)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Filtrar las mejores coincidencias (15%)
    num_good_matches = max(4, int(len(matches) * 0.15))  # Se requieren al menos 4 coincidencias
    good_matches = matches[:num_good_matches]
    
    if len(good_matches) < 4:
        print("No hay suficientes coincidencias para calcular la homografía.")
        exit(1)
    
    # Extraer las coordenadas de los puntos coincidentes de cada imagen
    pts_img1 = np.float32([ keypoints1[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
    pts_img2 = np.float32([ keypoints2[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
    
    # Calcular la homografía que mapea los puntos de img2 a img1 utilizando RANSAC
    H, status = cv2.findHomography(pts_img2, pts_img1, cv2.RANSAC, 5.0)
    if H is None:
        print("No se pudo calcular la homografía.")
        return None
    
    # -----------------------------------------------------
    # Crear el panorama combinando las dos imágenes
    # -----------------------------------------------------
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Calcular las esquinas de img2 y proyectarlas en img1 mediante la homografía
    corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners_img2, H)
    
    # Unir las esquinas de img1 y las transformadas de img2 para obtener las dimensiones del panorama
    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    all_corners = np.concatenate((corners_img1, transformed_corners), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Calcular la transformación de traslación para evitar coordenadas negativas
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]],
                              [0, 1, translation_dist[1]],
                              [0, 0, 1]])
    
    panorama_width = x_max - x_min
    panorama_height = y_max - y_min
    
    # Aplicar la homografía y la traslación a img2
    panorama = cv2.warpPerspective(img2, H_translation.dot(H), (panorama_width, panorama_height))
    
    # Copiar img1 en el lienzo del panorama
    panorama[translation_dist[1]:h1 + translation_dist[1], translation_dist[0]:w1 + translation_dist[0]] = img1
    
    return panorama

# ====================================================
# Paso 2: Crear el panorama iterativo a partir del array de imágenes
# ====================================================
# Se toma la primera imagen como base y se van uniendo las siguientes
panorama = images[0]

for i in range(1, len(images)):
    print(f"Uniendo imagen {i+1} de {len(images)}...")
    result = stitch_images(panorama, images[i])
    if result is None:
        print(f"Error al unir la imagen {i+1}. Se detiene el proceso.")
        break
    panorama = result

# ====================================================
# Paso 3: Visualización del panorama final
# ====================================================
if panorama is not None:
    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.title('Panorama Combinado')
    plt.axis('off')
    plt.show()
else:
    print("No se pudo crear el panorama.")
