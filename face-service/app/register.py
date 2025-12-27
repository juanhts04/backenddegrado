import face_recognition
import os
from fastapi import UploadFile, File, Form

DATASET_PATH = "dataset"

async def register_face(
    student_id: str = Form(...),
    file: UploadFile = File(...)
):
    # Leer imagen
    image = face_recognition.load_image_file(file.file)

    # Obtener codificaciones faciales
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        return {
            "ok": False,
            "message": "No se detectó ningún rostro"
        }

    # Crear carpeta si no existe
    os.makedirs(DATASET_PATH, exist_ok=True)

    # Guardar encoding
    file_path = f"{DATASET_PATH}/{student_id}.npy"
    encodings[0].dump(file_path)

    return {
        "ok": True,
        "student_id": student_id
    }
