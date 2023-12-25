const express = require('express');
const path = require('path');
const app = express();

app.use(express.json());

// 'build' 폴더를 정적 파일로 제공하도록 설정
app.use(express.static(path.join(__dirname, 'build')));

// 모든 경로에 대해 'index.html'을 반환
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

// POST 요청을 '/process_query' 경로로 처리
app.post('/process_query', (req, res) => {
  const query = req.body.query;
  // 쿼리 처리 로직
  // 예: const response = { response: "처리된 메시지", graphImages: ["image1.png", "image2.png"] };
  res.json(response);
});

// POST 요청을 '/get_graph_description' 경로로 처리
app.post('/get_graph_description', (req, res) => {
  const graphImageName = req.body.graphImageName;
  // 그래프 설명 처리 로직
  // 예: const description = { graphDescription: ["설명 1", "설명 2"] };
  res.json(description);
});


const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
