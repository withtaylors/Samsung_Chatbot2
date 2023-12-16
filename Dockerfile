# Node.js 이미지를 기반으로 합니다.
FROM node:14

# 작업 디렉토리를 설정합니다.
WORKDIR /app

# Node.js 종속성을 복사하고 설치합니다.
COPY package.json package-lock.json ./
RUN npm install

# React 소스 코드를 복사합니다.
COPY src ./src
COPY public ./public

# React 앱을 빌드합니다.
RUN npm run build

# 정적 파일을 서빙할 서버를 설치합니다.
RUN npm install -g serve

# 3000번 포트를 지정합니다.
EXPOSE 3000

# 빌드된 앱을 서빙합니다.
CMD ["serve", "-s", "build", "-l", "3000"]
