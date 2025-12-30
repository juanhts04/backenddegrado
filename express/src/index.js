const express = require("express");
const cors = require("cors");
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");

const app = express();
app.use(cors());
app.use(express.json());

// Multer: guardar imagen en memoria
const upload = multer({ storage: multer.memoryStorage() });

// Ruta de prueba
app.get("/", (req, res) => {
  res.json({ status: "Express API running" });
});

// 🚀 Ruta que recibe imagen desde Ionic
app.post("/recognize-face", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No se recibió imagen" });
    }

    // Crear form-data para FastAPI
    const formData = new FormData();
    formData.append("image", req.file.buffer, {
      filename: "face.jpg",
      contentType: req.file.mimetype,
    });

    // Enviar imagen a FastAPI (face-service)
    const response = await axios.post(
      "http://face-service:8000/recognize",
      formData,
      {
        headers: formData.getHeaders(),
      }
    );

    // Responder a Ionic
    res.json({
      message: "Express conectado a face-service",
      faceService: response.data,
    });

  } catch (error) {
    console.error("Error en reconocimiento:", error.message);
    res.status(500).json({ error: "Error procesando reconocimiento facial" });
  }
});

// Puerto
app.listen(3000, () => {
  console.log("Express running on port 3000");
});
