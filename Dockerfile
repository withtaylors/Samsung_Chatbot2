# 첫 번째 스테이지: Node.js 18 버전을 사용하여 React 앱을 빌드합니다.
FROM node:18 AS builder

# 작업 디렉토리를 React 앱 소스 코드 디렉토리로 설정합니다.
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# React 소스 코드만 복사합니다.
COPY . .

# React 앱을 빌드합니다.
RUN npm install && npm run build

# 두 번째 스테이지: Ubuntu 20.04 이미지를 기반으로 새로운 이미지를 시작합니다.
FROM ubuntu:20.04
WORKDIR /app

# 시간대를 설정합니다.
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 필수 패키지들을 설치합니다.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip git nodejs npm unzip

# pip를 최신 버전으로 업그레이드합니다.
RUN pip3 install --upgrade pip

# Python 패키지들을 설치합니다.
RUN pip3 install peft Faiss-cpu langchain bitsandbytes \
    git+https://github.com/huggingface/transformers.git \
    git+https://github.com/huggingface/peft.git \
    git+https://github.com/huggingface/accelerate.git \
    datasets rank_bm25 sentencepiece sentence-transformers \
    pypdf unstructured elasticsearch openai tiktoken Flask \
    flask-cors matplotlib

# Install http-server globally
RUN npm install -g http-server

# Copy the built React app from the builder stage
COPY --from=builder /app/build /app/build

# Expose the port that http-server will run on
EXPOSE 3000
EXPOSE 5000

# Serve the React app using http-server
CMD ["http-server", "/app/build", "-p", "3000"]

