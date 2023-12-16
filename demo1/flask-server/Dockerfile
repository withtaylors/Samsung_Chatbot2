# Python 이미지를 기반으로 합니다.
FROM python:3.8

# 작업 디렉토리를 설정합니다.
WORKDIR /app
# 필요한 패키지를 설치합니다.
RUN apt-get update && apt-get install -y python3-pip git

# 의존성 파일을 복사하고 설치합니다.
COPY demo1/flask-server/install.txt ./
RUN pip install --no-cache-dir -r install.txt

# Flask 애플리케이션의 나머지 부분을 복사합니다.
COPY demo1/flask-server .

# Flask 서버가 사용할 포트를 지정합니다.
EXPOSE 5000

# Flask 애플리케이션을 실행합니다.
CMD ["python", "samsung.py"]
