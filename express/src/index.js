const express = require("express");
const cors = require("cors");
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");

const app = express();
app.use(cors());
app.use(express.json());

// Multer en memoria (correcto)
const upload = multer({ storage: multer.memoryStorage() });

app.get("/", (req, res) => {
  res.json({ status: "Express API running" });
});

app.post("/recognize-face", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No se recibió imagen" });
    }

    console.log("Imagen recibida:", {
      name: req.file.originalname,
      type: req.file.mimetype,
      size: req.file.size
    });

    const formData = new FormData();
    formData.append("image", req.file.buffer, {
      filename: "face.jpg",
      contentType: req.file.mimetype,
    });

 const response = await axios.post(
  "https://juanbiometric.duckdns.org/recognize", // ✅ ahora apunta al subdominio HTTPS
  formData,
  { headers: formData.getHeaders() }
);

    console.log("Respuesta REAL FastAPI:", response.data);

    // 🔥 DEVOLVER SOLO LA RESPUESTA REAL
    res.json(response.data);

  } catch (error) {
    console.error("Error en reconocimiento:", error.response?.data || error.message);
    res.status(500).json({ error: "Error procesando reconocimiento facial" });
  }
});

app.listen(3000, () => {
  console.log("Express running on port 3000");
});
