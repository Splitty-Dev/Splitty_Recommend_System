# S3 로그 기반 증분 학습 시스템

S3에 저장된 사용자 행동 로그를 자동으로 가져와서 추천 모델을 재학습하는 시스템입니다.

## 📋 목차

1. [시스템 구조](#시스템-구조)
2. [설치 방법](#설치-방법)
3. [AWS 설정](#aws-설정)
4. [사용 방법](#사용-방법)
5. [로그 포맷](#로그-포맷)
6. [파이프라인 동작](#파이프라인-동작)

---

## 🏗️ 시스템 구조

```
S3 Bucket (로그 저장)
    ↓
s3_log_fetcher.py (로그 다운로드)
    ↓
log_data_transformer.py (학습 데이터 변환)
    ↓
data_merger.py (기존 데이터와 병합)
    ↓
incremental_training_pipeline.py (모델 재학습)
    ↓
saved_models/ (새 모델 저장)
```

### 주요 모듈

| 파일 | 역할 |
|------|------|
| `s3_log_fetcher.py` | S3에서 로그 파일을 다운로드 |
| `log_data_transformer.py` | JSON 로그를 학습 데이터로 변환 |
| `data_merger.py` | 새 데이터와 기존 데이터 병합 |
| `incremental_training_pipeline.py` | 전체 파이프라인 통합 실행 |

---

## 📦 설치 방법

### 1. Python 패키지 설치

```bash
pip install boto3 pandas numpy torch implicit scikit-learn
```

### 2. 파일 구조 확인

```
server_fastAPI/
├── s3_log_fetcher.py
├── log_data_transformer.py
├── data_merger.py
├── incremental_training_pipeline.py
├── hybrid_recommender.py
└── saved_models/
```

---

## 🔐 AWS 설정

### 1. AWS Credentials 설정

**옵션 A: AWS CLI 사용 (권장)**
```bash
aws configure
# AWS Access Key ID: <your-key>
# AWS Secret Access Key: <your-secret>
# Default region: ap-northeast-2
```

**옵션 B: 환경 변수 설정**
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="ap-northeast-2"
```

**옵션 C: ~/.aws/credentials 파일**
```ini
[default]
aws_access_key_id = your-access-key
aws_secret_access_key = your-secret-key
region = ap-northeast-2
```

### 2. S3 버킷 권한 확인

IAM 사용자에게 다음 권한이 필요합니다:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::splitty-recommendation-log-bucket",
        "arn:aws:s3:::splitty-recommendation-log-bucket/*"
      ]
    }
  ]
}
```

---

## 🚀 사용 방법

### 기본 사용법

```bash
cd server_fastAPI
python3 incremental_training_pipeline.py
```

### S3 버킷 구조별 사용법

#### 1. 루트 디렉토리에서 검색 (기본)
```bash
python3 incremental_training_pipeline.py
# splitty-recommendation-log-bucket의 모든 .json 파일 검색
```

#### 2. 특정 경로에서 검색
```bash
# logs/ 디렉토리 하위 모든 파일 검색
python3 incremental_training_pipeline.py --prefix logs/

# 2025년 11월 로그만 검색
python3 incremental_training_pipeline.py --prefix logs/2025/11/

# 특정 날짜 로그만 검색
python3 incremental_training_pipeline.py --prefix logs/2025/11/01/
```

#### 3. 버킷 구조 예시
```
splitty-recommendation-log-bucket/
├── logs/
│   ├── 2025/
│   │   ├── 11/
│   │   │   ├── 01/
│   │   │   │   ├── user_actions_001.json  ✓ 자동 감지
│   │   │   │   └── user_actions_002.json  ✓ 자동 감지
│   │   │   └── 02/
│   │   │       └── user_actions_003.json  ✓ 자동 감지
│   │   └── 10/
│   └── archive/
│       └── old_logs.json                   ✓ 자동 감지
└── user_actions.json                       ✓ 자동 감지

# prefix="logs/2025/11/" 사용 시
→ logs/2025/11/ 하위의 모든 .json 파일을 재귀적으로 검색
```

### 옵션 사용

```bash
# 로그 파일 최대 5개만 가져오기
python3 incremental_training_pipeline.py --max-files 5

# 특정 경로의 로그만 가져오기
python3 incremental_training_pipeline.py --prefix logs/2025/11/

# 경로 + 파일 개수 제한
python3 incremental_training_pipeline.py --prefix logs/2025/11/ --max-files 5

# 데이터 병합만 하고 모델 재학습 스킵
python3 incremental_training_pipeline.py --no-retrain

# 기존 데이터 백업 스킵
python3 incremental_training_pipeline.py --no-backup

# 다른 S3 버킷 사용
python3 incremental_training_pipeline.py --bucket my-other-bucket

# 데이터 디렉토리 지정
python3 incremental_training_pipeline.py --data-dir ../data/my_data

# 모델 저장 경로 지정
python3 incremental_training_pipeline.py --model-dir ./my_models
```

### 모든 옵션 보기

```bash
python3 incremental_training_pipeline.py --help
```

---

## 📝 로그 포맷

### S3에 저장되는 로그 형식

```json
[
  {
    "timestamp": 1762003990140,
    "item_id": 31,
    "user_id": "1",
    "action": "VIEW",
    "category_id": 1,
    "price": 20000
  },
  {
    "timestamp": 1762003995853,
    "item_id": 45,
    "user_id": "2",
    "action": "PURCHASE",
    "category_id": 3,
    "price": 10000
  }
]
```

### 필드 설명

| 필드 | 타입 | 설명 | 예시 |
|------|------|------|------|
| `timestamp` | int | 밀리초 단위 타임스탬프 | 1762003990140 |
| `item_id` | int | 아이템 ID | 31 |
| `user_id` | string | 사용자 ID | "1" |
| `action` | string | 사용자 행동 (VIEW, CLICK, ADD_CART, PURCHASE) | "VIEW" |
| `category_id` | int | 카테고리 ID (1-6) | 1 |
| `price` | int | 아이템 가격 | 20000 |

### 액션 타입별 가중치

| Action | Weight | 설명 |
|--------|--------|------|
| VIEW | 1.0 | 아이템 조회 |
| CLICK | 2.0 | 아이템 클릭 |
| ADD_CART | 3.0 | 장바구니 추가 |
| PURCHASE | 5.0 | 구매 (가장 중요) |

---

## ⚙️ 파이프라인 동작

### 중복 방지 메커니즘

파이프라인은 **이미 처리된 로그 파일을 자동으로 스킵**합니다:

```python
# processed_log_files.json에 처리된 파일 목록 저장
{
  "processed_files": [
    "logs/2025/11/01/user_actions_001.json",
    "logs/2025/11/01/user_actions_002.json"
  ],
  "last_updated": "2025-11-01T03:05:23"
}
```

- ✅ 처리된 파일은 다시 가져오지 않음
- ✅ S3 API 호출 최소화
- ✅ 중복 학습 데이터 방지

처리 기록을 초기화하려면:
```bash
rm processed_log_files.json
```

### 1단계: S3 로그 가져오기
```python
fetcher = S3LogFetcher("splitty-recommendation-log-bucket")
logs = fetcher.fetch_latest_logs(max_files=10, skip_processed=True)
# 출력: 발견된 로그 파일: 30개
#       새로운 로그 1,500개를 가져왔습니다.
#       처리된 파일 스킵: 20개
#       새로 가져온 파일: 10개
```

### 2단계: 학습 데이터로 변환
```python
transformer = LogDataTransformer()
training_data = transformer.transform_logs_to_training_data(logs)
# 출력: 로그 변환 시작: 1,500개 레코드
#       변환 완료: 1,350개 학습 레코드
#       유니크 사용자: 120명
#       유니크 아이템: 300개
```

### 3단계: 기존 데이터와 병합
```python
merger = DataMerger("../data/splitty_recommendation_data_1")
train_df, val_df, test_df = merger.load_existing_data()
merged = merger.merge_new_data(train_df, training_data, merge_strategy="train_only")
# 출력: 기존 train: 14,000개
#       새 데이터: 1,350개
#       병합 후: 15,200개
#       중복 제거: 150개 제거됨
#       최종 train: 15,050개
#       Val: 3,000개 (유지)
#       Test: 3,000개 (유지)
```

### 4단계: 데이터 저장
```python
merger.save_merged_data(merged_train, val_df, test_df, backup=True)
# 출력: Train 데이터만 업데이트, Val/Test는 유지
```

### 5단계: 모델 재학습
```python
recommender = HybridRecommender()
recommender.load_data("../data/splitty_recommendation_data_1")
recommender.train_models(mf_factors=50, epochs=30, batch_size=512)
recommender.save_models("./saved_models")
# 출력: 모델 학습 완료
#       새 모델 저장: ./saved_models
```

---

## 🔄 주기적 실행 설정

### cron을 사용한 자동화 (리눅스/맥)

```bash
# crontab 편집
crontab -e

# 매일 새벽 3시에 실행
0 3 * * * cd /path/to/server_fastAPI && python3 incremental_training_pipeline.py >> /path/to/logs/training.log 2>&1

# 매주 일요일 새벽 2시에 실행
0 2 * * 0 cd /path/to/server_fastAPI && python3 incremental_training_pipeline.py >> /path/to/logs/training.log 2>&1
```

### systemd 타이머 사용 (리눅스)

```ini
# /etc/systemd/system/incremental-training.service
[Unit]
Description=Incremental Training Pipeline
After=network.target

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=/home/ubuntu/Splitty_Recommend_System/server_fastAPI
ExecStart=/usr/bin/python3 incremental_training_pipeline.py

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/incremental-training.timer
[Unit]
Description=Daily Incremental Training
Requires=incremental-training.service

[Timer]
OnCalendar=daily
OnCalendar=03:00
Persistent=true

[Install]
WantedBy=timers.target
```

활성화:
```bash
sudo systemctl enable incremental-training.timer
sudo systemctl start incremental-training.timer
sudo systemctl status incremental-training.timer
```

---

## 📊 모니터링

### 로그 확인

파이프라인 실행 시 자세한 로그가 출력됩니다:

```
================================================================================
증분 학습 파이프라인 시작
시작 시간: 2025-11-01 03:00:00
================================================================================

[1/6] S3에서 로그 가져오기...
발견된 로그 파일: 30개
새로운 로그 1,500개를 가져왔습니다.
  처리된 파일 스킵: 20개
  새로 가져온 파일: 10개
✓ 1,500개의 로그 레코드를 가져왔습니다.

[2/6] 로그를 학습 데이터로 변환...
로그 변환 시작: 1,500개 레코드
액션 분포:
VIEW        800
CLICK       400
PURCHASE    200
ADD_CART    100
✓ 1,350개의 학습 레코드로 변환되었습니다.

[3/6] 기존 학습 데이터 로드...
Train: 14000개, Test: 3000개, Val: 3000개
✓ 기존 데이터 로드 완료

[4/6] 새 데이터를 train에 추가 (test/val은 유지)...
데이터 병합 시작 (전략: train_only)...
  기존 train: 14000개
  새 데이터: 1350개
  병합 후: 15350개
  중복 제거 중...
  중복 제거: 300개 제거됨
최종 train: 15050개

[5/6] 새 train 데이터 저장 (test/val 유지)...
  백업: ../data/splitty_recommendation_data_1/user_item_train.csv_20251101_030005.backup
✓ 데이터 저장 완료
  최종 - Train: 15050개, Val: 3000개, Test: 3000개

[6/6] 모델 재학습...
  하이브리드 추천 모델 초기화...
  데이터 로드 중...
  모델 학습 중... (이 과정은 시간이 걸릴 수 있습니다)
  Epoch 1/30, Loss: 0.4523
  Epoch 2/30, Loss: 0.3891
  ...
  기존 모델 백업: ./saved_models_20251101_030500.backup
  새 모델 저장: ./saved_models
✓ 모델 재학습 완료

================================================================================
증분 학습 파이프라인 완료!
종료 시간: 2025-11-01 03:10:23
================================================================================
```

---

## 🐛 문제 해결

### AWS 인증 오류
```
ClientError: An error occurred (AccessDenied) when calling the GetObject operation
```
→ AWS credentials 확인, IAM 권한 확인

### 버킷을 찾을 수 없음
```
NoSuchBucket: The specified bucket does not exist
```
→ S3 버킷 이름 확인, 리전 확인

### 메모리 부족
```
MemoryError: Unable to allocate array
```
→ `--max-files` 옵션으로 로그 파일 개수 줄이기

---

## 📞 문의

문제가 발생하면 다음을 확인하세요:
1. AWS credentials 설정
2. S3 버킷 접근 권한
3. Python 패키지 설치 상태
4. 로그 파일 포맷

---

**작성일**: 2025-11-01  
**버전**: 1.0.0
