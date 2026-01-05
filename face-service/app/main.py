from fastapi import FastAPI, UploadFile, File, Form
import face_recognition
import numpy as np
import os
from fastapi.openapi.utils import get_openapi

app = FastAPI(
    title="Face Recognition API",
    description="Servicio de reconocimiento facial",
    version="1.0.0",
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["servers"] = [
        {"url": "https://juanbiometric.duckdns.org", "description": "Producción"},
        {"url": "http://localhost:8000", "description": "Local"},
    ]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

FACES_DIR = "faces"
os.makedirs(FACES_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"status": "Face service running"}

@app.post("/register-face")
async def register_face(
    name: str = Form(...),
    image: UploadFile = File(...)
):
    img = face_recognition.load_image_file(image.file)
    encodings = face_recognition.face_encodings(img)

    if not encodings:
        return {"error": "No se detectó ningún rostro"}

    np.save(f"{FACES_DIR}/{name}.npy", encodings[0])

    return {
        "message": "Rostro registrado",
        "person": name
    }

@app.post("/recognize")
async def recognize_face(image: UploadFile = File(...)):
    # 1️⃣ Leer imagen
    img = face_recognition.load_image_file(image.file)
    encodings = face_recognition.face_encodings(img)

    if not encodings:
        return {
            "recognized": False,
            "message": "No se detectó ningún rostro"
        }

    unknown_encoding = encodings[0]

    # 2️⃣ Cargar rostros registrados (.npy)
    known_encodings = []
    known_names = []

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

    # 3️⃣ Comparar
    distances = face_recognition.face_distance(
        known_encodings,
        unknown_encoding
    )

    best_match = np.argmin(distances)

    if distances[best_match] < 0.45:
        return {
            "recognized": True,
            "person": known_names[best_match],
            "confidence": float(1 - distances[best_match])
        }

    return {
        "recognized": False,
        "message": "Rostro no reconocido"
    }