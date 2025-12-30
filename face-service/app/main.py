from fastapi import FastAPI, UploadFile, File
from recognize import recognize_face

app = FastAPI()

@app.get("/")
def root():
    return {"status": "Face service running"}

@app.post("/recognize")
async def recognize(image: UploadFile = File(...)):
    image_bytes = await image.read()

    result = recognize_face(image_bytes)

    return result
