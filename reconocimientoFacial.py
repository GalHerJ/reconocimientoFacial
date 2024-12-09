import face_recognition
import cv2 

def load_and_encode_image(image_path):
    # Cargar la imagen
    image = face_recognition.load_image_file(image_path)
    
    # Obtener las codificaciones de los rostros en la imagen
    face_encodings = face_recognition.face_encodings(image)

    if not face_encodings:
        raise ValueError(f"No se detecta ningún rostro en {image_path}")

    # Devolver la primera codificación (puedes ajustar esto si tienes múltiples rostros)
    return face_encodings[0]

def compare_faces(image1_path, image2_path):
    try:
        # Obtener las codificaciones de las dos imágenes
        encoding1 = load_and_encode_image(image1_path)
        encoding2 = load_and_encode_image(image2_path) 

        # Comparar las codificaciones de los dos rostros
        results = face_recognition.compare_faces([encoding1], encoding2)
        distance = face_recognition.face_distance([encoding1], encoding2)

        # Mostrar los resultados de la comparación
        if results[0]:
            return f"Coincidencia encontrada: {distance[0]:.2f}"
        else:
            return f"No hay coincidencia: {distance[0]:.2f}"

    except Exception as e:
        return str(e)

# Asegúrate de usar "__main__" para la ejecución directa del script
if __name__ == "__main__":
    image1_path = "/Users/josemanuelgalvanhernandez/Descargas/e65b3aeb-2cb6-4868-ac4f-b179bc647855.JPG"
    image2_path = "/Users/josemanuelgalvanhernandez/Descargas/329499de-c134-4945-bdc8-11529006aa47.JPG"

    results = compare_faces(image1_path, image2_path)
    print(results)


