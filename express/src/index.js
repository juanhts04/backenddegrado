const express = require("express");
const cors = require("cors");

const app = express();
app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
  res.json({ status: "Express API running" });
});

app.listen(3000, () => {
  console.log("Express running on port 3000");
});
