# 하이퍼파라미터 튜닝 가이드

## 📋 개요

Validation set을 사용하여 Grid Search로 최적의 하이퍼파라미터를 찾는 스크립트입니다.

## 🎯 평가 지표

다음 4가지 지표로 모델 성능을 평가합니다:

1. **Precision@K**: 추천한 K개 아이템 중 실제로 좋아한 비율
2. **Recall@K**: 좋아할 아이템 중 추천한 K개에 포함된 비율
3. **NDCG@K**: 순위를 고려한 추천 품질 (0~1, 높을수록 좋음)
4. **Hit Rate@K**: 최소 1개라도 맞춘 사용자 비율

주요 지표: **NDCG@10** (최적화 목표)

## 🔧 튜닝할 하이퍼파라미터

### 1. `mf_factors` (Matrix Factorization 잠재 요인 수)
- **범위**: [30, 50, 70]
- **의미**: 사용자/아이템 임베딩 벡터의 차원
- **효과**: 
  - 작을수록: 빠르지만 표현력 낮음
  - 클수록: 느리지만 복잡한 패턴 포착 가능

### 2. `epochs` (Two-Tower 학습 에포크)
- **범위**: [20, 30, 40]
- **의미**: Two-Tower 모델을 몇 번 반복 학습할지
- **효과**:
  - 작을수록: 빠르지만 과소적합 가능
  - 클수록: 느리고 과적합 위험

### 3. `batch_size` (배치 크기)
- **범위**: [256, 512, 1024]
- **의미**: 한 번에 처리할 데이터 개수
- **효과**:
  - 작을수록: 메모리 적게 사용, 학습 불안정
  - 클수록: 메모리 많이 사용, 학습 안정적

### 4. `top_k` (MF 후보 개수)
- **범위**: [150, 250, 350]
- **의미**: Matrix Factorization에서 몇 개 후보를 뽑을지
- **효과**:
  - 작을수록: 빠르지만 좋은 아이템 놓칠 수 있음
  - 클수록: 느리지만 더 많은 후보 고려

### 5. `top_n` (최종 추천 개수)
- **고정값**: 50
- **의미**: 최종적으로 사용자에게 추천할 개수

## 🚀 사용 방법

### Option 1: 빠른 테스트 (권장 - 처음 사용 시)
```bash
cd server_fastAPI
python quick_tuning.py
```
- **조합 수**: 8가지
- **예상 시간**: 20-30분
- **용도**: 빠르게 결과 확인

### Option 2: 전체 Grid Search
```bash
cd server_fastAPI
python hyperparameter_tuning.py
```
- **조합 수**: 81가지 (3×3×3×3×1)
- **예상 시간**: 3-4시간
- **용도**: 최적 파라미터 찾기

## 📊 결과 확인

### 1. 실시간 진행 상황
```
실험 5/81
파라미터:
  mf_factors: 50
  epochs: 30
  batch_size: 512
  top_k: 250
  top_n: 50

평가 결과:
  Precision@5: 0.0234
  Precision@10: 0.0189
  Recall@5: 0.0456
  Recall@10: 0.0823
  NDCG@5: 0.0312
  NDCG@10: 0.0401
  HitRate@5: 0.1234
  HitRate@10: 0.2156

🎉 새로운 최고 점수! NDCG@10: 0.0401
```

### 2. 최종 결과
```
🏆 최적 하이퍼파라미터:
  mf_factors: 50
  epochs: 30
  batch_size: 512
  top_k: 250
  top_n: 50

📊 최고 성능 지표 (NDCG@10: 0.0401):
  Precision@5: 0.0234
  Precision@10: 0.0189
  Recall@5: 0.0456
  Recall@10: 0.0823
  NDCG@5: 0.0312
  NDCG@10: 0.0401
  HitRate@5: 0.1234
  HitRate@10: 0.2156
```

### 3. 저장되는 파일

#### `saved_models/best_model/`
- `matrix_factorization.pkl`: 최고 성능 MF 모델
- `two_tower_model.pth`: 최고 성능 Two-Tower 모델

#### `saved_models/best_params.json`
```json
{
  "params": {
    "mf_factors": 50,
    "epochs": 30,
    "batch_size": 512,
    "top_k": 250,
    "top_n": 50
  },
  "metrics": {
    "Precision@5": 0.0234,
    "Precision@10": 0.0189,
    "Recall@5": 0.0456,
    "Recall@10": 0.0823,
    "NDCG@5": 0.0312,
    "NDCG@10": 0.0401,
    "HitRate@5": 0.1234,
    "HitRate@10": 0.2156
  },
  "composite_score": 0.0401
}
```

#### `saved_models/grid_search_results_YYYYMMDD_HHMMSS.csv`
모든 실험 결과가 저장된 CSV 파일 (Excel로 열어서 분석 가능)

## 📈 결과 해석

### 좋은 성능 기준 (추천 시스템 일반적 기준)
- **NDCG@10**: 0.03~0.05 (보통), 0.05~0.10 (좋음), 0.10+ (매우 좋음)
- **Precision@10**: 0.01~0.03 (보통), 0.03~0.05 (좋음), 0.05+ (매우 좋음)
- **Hit Rate@10**: 0.10~0.20 (보통), 0.20~0.40 (좋음), 0.40+ (매우 좋음)

### 성능이 낮다면?
1. **데이터 문제**: 상호작용 데이터가 부족하거나 희소함
2. **모델 문제**: 더 복잡한 모델 필요
3. **파라미터 문제**: 더 넓은 범위로 Grid Search 필요
4. **피처 문제**: 추가 피처 필요 (시간, 위치 등)

## 🔄 최고 모델 적용

### 1. FastAPI 서버에서 사용
`main.py`를 수정하여 best_model 로드:
```python
# main.py의 load_hybrid_recommender() 함수에서
model_path = os.path.join(current_dir, "saved_models", "best_model")
hybrid_recommender.load_models(model_path, data_path)
```

### 2. 서버 재시작
```bash
# EC2에서
sudo systemctl restart splitty-recommend
```

## ⚙️ 커스터마이징

### Grid 범위 변경
`hyperparameter_tuning.py`의 `main()` 함수에서:
```python
param_grid = {
    'mf_factors': [20, 40, 60, 80],      # 더 많은 옵션
    'epochs': [15, 25, 35, 45],          # 더 많은 옵션
    'batch_size': [128, 256, 512, 1024], # 더 많은 옵션
    'top_k': [100, 200, 300, 400],       # 더 많은 옵션
    'top_n': [50],
}
```

### 평가 K 값 변경
```python
eval_k_list = [3, 5, 10, 20, 50]  # 더 다양한 K 값
```

## 🐛 문제 해결

### 메모리 부족
```python
param_grid = {
    'batch_size': [128, 256],  # 더 작은 배치
    'mf_factors': [30, 40],    # 더 작은 차원
}
```

### 너무 느림
```python
param_grid = {
    'epochs': [10, 20],        # 에포크 줄이기
    'top_k': [100, 150],       # 후보 줄이기
}
```

### Validation 데이터 없음
```bash
# 데이터 생성 스크립트 실행
cd data/generation
jupyter notebook splitty_recommendation_data_generation.ipynb
```

## 📝 참고사항

1. **GPU 사용**: `device='cuda'`로 변경하면 더 빠름 (GPU 있는 경우)
2. **중단 후 재개**: 현재는 지원 안 함 (처음부터 다시 실행)
3. **병렬 처리**: 현재는 순차 처리 (추후 업데이트 예정)
4. **교차 검증**: 현재는 단일 validation set (추후 K-fold 추가 예정)

## 🎓 추가 학습 자료

- [추천 시스템 평가 지표](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
- [NDCG 이해하기](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)
- [하이퍼파라미터 튜닝 전략](https://scikit-learn.org/stable/modules/grid_search.html)
