import cv2

# Cargar la imagen
imagen = cv2.imread("./Mariposa.jpg")

# Convertir la imagen a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Establecer el umbral de Hessian a 500
# Crear un objeto SURF
surf = cv2.xfeatures2d.SURF_create(500)

# Detectar keypoints y descriptores
keypoints, descriptores = surf.detectAndCompute(gris, None)

# Dibujar los keypoints en la imagen
imagen_keypoints = cv2.drawKeypoints(imagen, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Mostrar la imagen con los keypoints
cv2.imshow("Keypoints SIFT", imagen_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()