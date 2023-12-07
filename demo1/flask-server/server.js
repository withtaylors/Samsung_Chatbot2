const express = require("express");
const { spawn } = require("child_process");
const app = express();
const port = 5000;

// Body parsing Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.post("/process_query", (req, res) => {
  const message = req.body.message;
  const pythonProcess = spawn("python3", ["/app/src/samsung.py", message]);

  pythonProcess.stdout.on("data", (data) => {
    console.log(`Received data from Python script: ${data}`);
    res.send(data.toString());
  });

  pythonProcess.on("close", (code) => {
    console.log(`Python script process exited with code ${code}`);
  });

  console.log(`Received message: ${message}`);
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
