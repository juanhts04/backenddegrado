from fastapi import FastAPI
from register import register_face
from recognize import recognize_face

app = FastAPI(title="Face Recognition Service")

@app.get("/")
def root():
    return {"status": "Face service running"}

app.post("/register")(register_face)
app.post("/recognize")(recognize_face)
