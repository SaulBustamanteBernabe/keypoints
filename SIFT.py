import cv2

# Cargar la imagen
imagen = cv2.imread("./Mariposa.jpg")

# Convertir la imagen a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Crear un objeto SIFT
sift = cv2.SIFT_create()

# Detectar keypoints y descriptores
keypoints, descriptores = sift.detectAndCompute(gris, None)

# Dibujar los keypoints en la imagen
imagen_keypoints = cv2.drawKeypoints(imagen, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Mostrar la imagen con los keypoints
cv2.imshow("Keypoints SIFT", imagen_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()