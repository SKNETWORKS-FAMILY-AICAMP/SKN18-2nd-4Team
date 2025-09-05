FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사
COPY . .

# 로그 디렉토리 생성
RUN mkdir -p logs outputs

# 실행 권한 부여
RUN chmod +x run_training.py run_prediction.py

# 기본 명령어
CMD ["python", "run_training.py"]
