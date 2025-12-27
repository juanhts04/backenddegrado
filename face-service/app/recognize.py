import face_recognition
import numpy as np
import os
from fastapi import UploadFile, File

DATASET_PATH = "dataset"

async def recognize_face(file: UploadFile = File(...)):
    # Leer imagen
    image = face_recognition.load_image_file(file.file)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        return {"match": False}

    unknown_encoding = encodings[0]

    # Recorrer dataset
    for filename in os.listdir(DATASET_PATH):
        if not filename.endswith(".npy"):
            continue

        known_encoding = np.load(
            os.path.join(DATASET_PATH, filename),
            allow_pickle=True
        )

        matches = face_recognition.compare_faces(
            [known_encoding],
            unknown_encoding,
            tolerance=0.5
        )

        if matches[0]:
            return {
                "match": True,
                "student_id": filename.replace(".npy", "")
            }

    return {"match": False}
