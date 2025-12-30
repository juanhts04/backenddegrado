from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.get("/")
def root():
    return {"status": "Face service running"}

@app.post("/recognize")
def recognize_face(image_bytes: bytes):
    # Por ahora simulamos reconocimiento
    return {
        "recognized": True,
        "confidence": 0.95
    }
