import face_recognition
import numpy as np
import os
from io import BytesIO

FACES_DIR = "faces"

def recognize_face(image_bytes: bytes):
    # Cargar imagen enviada
    image = face_recognition.load_image_file(BytesIO(image_bytes))
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        return {
            "recognized": False,
            "message": "No se detectó ningún rostro"
        }

    unknown_encoding = encodings[0]

    known_encodings = []
    known_names = []

    # Cargar encodings guardados (.npy)
    for file in os.listdir(FACES_DIR):
        if file.endswith(".npy"):
            name = file.replace(".npy", "")
            encoding = np.load(os.path.join(FACES_DIR, file))
            known_encodings.append(encoding)
            known_names.append(name)

    if not known_encodings:
        return {
            "recognized": False,
            "message": "No hay rostros registrados"
        }

    matches = face_recognition.compare_faces(
        known_encodings,
        unknown_encoding,
        tolerance=0.45
    )

    distances = face_recognition.face_distance(
        known_encodings,
        unknown_encoding
    )

    best_match = np.argmin(distances)

    if matches[best_match]:
        confidence = float(1 - distances[best_match])
        return {
            "recognized": True,
            "person": known_names[best_match],
            "confidence": round(confidence, 2)
        }

    return {
        "recognized": False,
        "message": "Rostro no reconocido"
    }
