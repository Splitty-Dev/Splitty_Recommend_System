from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional
from hybrid_recommender import HybridRecommender

app = FastAPI()

# 요청 모델 정의
class RecommendRequest(BaseModel):
    user_id: str = Field(..., description="사용자 ID")
    top_n: int = Field(50, ge=1, le=100, description="상위 추천 개수")
    categoryId: Optional[int] = Field(None, ge=1, le=6, description="카테고리 ID (1-6: 1=식품, 2=생활/주방, 3=뷰티/미용, 4=패션, 5=건강/운동, 6=유아)")
    available_items: List[int] = Field(..., description="거리 내 사용 가능한 아이템 ID 리스트 (필수)")
    rank: Optional[int] = Field(None, ge=1, description="페이징 시작 순위 (1부터 시작, 기본값: 1)")

# 글로벌 변수로 데이터 저장
item_embeddings = None
encoders = None
user_item_test = None
user_item_val = None
data_loaded = False

# 하이브리드 추천 시스템
hybrid_recommender = None
hybrid_model_loaded = False

def simplify_response(recommendations: List[Dict]) -> List[Dict]:
    """추천 결과를 item_id와 rank만 포함하도록 간소화"""
    return [
        {
            "item_id": rec.get("item_id"),
            "rank": rec.get("rank", i + 1)
        }
        for i, rec in enumerate(recommendations)
    ]

def check_user_exists(user_id: str) -> bool:
    """사용자가 학습 데이터에 존재하는지 확인"""
    try:
        if hybrid_model_loaded and hybrid_recommender:
            # train_data에서 사용자 존재 확인 (user_idx 컬럼 사용)
            if hybrid_recommender.train_data is not None:
                return user_id in hybrid_recommender.train_data['user_idx'].astype(str).values
        
        # 폴백: 기존 데이터에서 확인
        if data_loaded and user_item_test is not None:
            if 'user_id' in user_item_test.columns:
                return user_id in user_item_test['user_id'].astype(str).values
            elif 'user_idx' in user_item_test.columns:
                return user_id in user_item_test['user_idx'].astype(str).values
        
        return False
    except Exception as e:
        print(f"사용자 확인 중 오류: {str(e)}")
        return False

def get_popular_recommendations_internal(top_n: int, category_filter: Optional[int] = None, available_items: Optional[List[int]] = None) -> List[Dict]:
    """내부용 인기 아이템 추천 함수 (category_filter: 1-6 정수)"""
    try:
        if hybrid_model_loaded and hybrid_recommender:
            recommendations = hybrid_recommender.get_popular_recommendations(
                top_n=top_n,
                category_filter=category_filter,
                available_items=available_items
            )
            return simplify_response(recommendations)
        
        # 폴백: 빈 리스트 반환
        return []
    except Exception as e:
        print(f"인기 아이템 추천 오류: {str(e)}")
        return []

def load_hybrid_recommender():
    """하이브리드 추천 시스템을 로드합니다."""
    global hybrid_recommender, hybrid_model_loaded
    
    if hybrid_model_loaded:
        return
    
    try:
        # 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        data_path = os.path.join(project_root, "data", "splitty_recommendation_data_1")
        model_path = os.path.join(current_dir, "saved_models")
        
        # 하이브리드 추천 시스템 초기화
        hybrid_recommender = HybridRecommender(device='cpu')
        
        # 저장된 모델이 있는지 확인
        if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "matrix_factorization.pkl")):
            print("기존 학습된 모델을 로드합니다...")
            hybrid_recommender.load_models(model_path, data_path)
        else:
            print("새로 모델을 학습합니다...")
            # 데이터 로드
            hybrid_recommender.load_data(data_path)
            
            # 모델 학습
            hybrid_recommender.train_models(
                mf_factors=50,
                epochs=30,  # 서버 시작 시간 단축을 위해 줄임
                batch_size=512
            )
            
            # 모델 저장
            hybrid_recommender.save_models(model_path)
        
        hybrid_model_loaded = True
        print("하이브리드 추천 시스템 로드 완료!")
        
    except Exception as e:
        print(f"하이브리드 추천 시스템 로드 실패: {str(e)}")
        # 기존 방식으로 폴백
        load_recommendation_data()

def load_recommendation_data():
    """기존 추천 시스템에 필요한 데이터를 로드합니다. (폴백용)"""
    global item_embeddings, encoders, user_item_test, user_item_val, data_loaded
    
    if data_loaded:
        return
    
    try:
        # 데이터 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        data_path = os.path.join(project_root, "data", "splitty_recommendation_data_1")
        
        # 아이템 임베딩 로드
        embedding_path = os.path.join(data_path, "item_title_embeddings.npz")
        if os.path.exists(embedding_path):
            embeddings_data = np.load(embedding_path)
            item_embeddings = embeddings_data['embeddings']
            print(f"Item embeddings loaded: {item_embeddings.shape}")
        else:
            raise FileNotFoundError(f"Embeddings file not found: {embedding_path}")
        
        # 인코더 로드
        encoders_path = os.path.join(data_path, "encoders.json")
        if os.path.exists(encoders_path):
            with open(encoders_path, 'r', encoding='utf-8') as f:
                encoders = json.load(f)
            print(f"Encoders loaded: {list(encoders.keys())}")
        else:
            raise FileNotFoundError(f"Encoders file not found: {encoders_path}")
        
        # 테스트 데이터 로드
        test_path = os.path.join(data_path, "user_item_test.csv")
        if os.path.exists(test_path):
            user_item_test = pd.read_csv(test_path)
            print(f"Test data loaded: {user_item_test.shape}")
        
        # 검증 데이터 로드
        val_path = os.path.join(data_path, "user_item_val.csv")
        if os.path.exists(val_path):
            user_item_val = pd.read_csv(val_path)
            print(f"Validation data loaded: {user_item_val.shape}")
        
        data_loaded = True
        print("모든 추천 데이터가 성공적으로 로드되었습니다.")
        
    except Exception as e:
        print(f"데이터 로딩 중 오류 발생: {str(e)}")
        raise e

def get_user_recommendations(user_id: str, top_n: int = 5) -> List[Dict]:
    """
    사용자에게 아이템을 추천합니다.
    콘텐츠 기반 필터링을 사용하여 사용자가 이전에 상호작용한 아이템과 유사한 아이템을 추천합니다.
    """
    global item_embeddings, encoders, user_item_test, user_item_val
    
    if not data_loaded:
        raise HTTPException(status_code=500, detail="추천 데이터가 로드되지 않았습니다.")
    
    # 사용자 데이터 결합 (test + validation)
    all_user_data = pd.concat([user_item_test, user_item_val], ignore_index=True) if user_item_val is not None else user_item_test
    
    # 해당 사용자의 상호작용 데이터 필터링
    user_interactions = all_user_data[all_user_data['user_idx'] == user_id]
    
    if user_interactions.empty:
        # 새 사용자인 경우 인기 있는 아이템 추천 (가중치 기반)
        if 'weight' in all_user_data.columns:
            popular_items = all_user_data.groupby('item_idx')['weight'].sum().sort_values(ascending=False).head(top_n)
        else:
            # 가중치가 없으면 빈도 기반
            popular_items = all_user_data['item_idx'].value_counts().head(top_n)
        
        recommendations = []
        for item_idx in popular_items.index:
            item_data = all_user_data[all_user_data['item_idx'] == item_idx].iloc[0]
            recommendations.append({
                'item_id': item_idx,
                'title': item_data.get('title', 'Unknown'),
                'category': item_data.get('category', 'Unknown'),
                'price': int(item_data.get('price', 0)),
                'score': float(popular_items[item_idx]),
                'reason': 'popular_item'
            })
        return recommendations
    
    # 사용자가 상호작용한 아이템들의 임베딩 평균 계산
    user_item_indices = []
    for item_idx in user_interactions['item_idx'].unique():
        if item_idx < len(item_embeddings):
            user_item_indices.append(item_idx)
    
    if not user_item_indices:
        # 임베딩을 찾을 수 없는 경우 인기 아이템 반환
        popular_items = all_user_data.groupby('item_idx').size().sort_values(ascending=False).head(top_n)
        recommendations = []
        for item_idx in popular_items.index:
            item_data = all_user_data[all_user_data['item_idx'] == item_idx].iloc[0]
            recommendations.append({
                'item_id': item_idx,
                'title': item_data.get('title', 'Unknown'),
                'category': item_data.get('category', 'Unknown'),
                'price': int(item_data.get('price', 0)),
                'score': float(popular_items[item_idx]),
                'reason': 'fallback_popular'
            })
        return recommendations
    
    # 사용자 프로필 벡터 (상호작용한 아이템들의 평균 임베딩)
    user_profile = np.mean(item_embeddings[user_item_indices], axis=0).reshape(1, -1)
    
    # 모든 아이템과의 유사도 계산
    similarities = cosine_similarity(user_profile, item_embeddings)[0]
    
    # 사용자가 이미 상호작용한 아이템 제외
    interacted_indices = set(user_item_indices)
    
    # 유사도 기반으로 상위 아이템 선택
    item_scores = []
    for idx, similarity in enumerate(similarities):
        if idx not in interacted_indices and similarity > 0:
            # 인덱스를 item_id로 변환
            item_id = None
            if 'idx_to_item_id' in encoders and str(idx) in encoders['idx_to_item_id']:
                item_id = encoders['idx_to_item_id'][str(idx)]
            elif 'item_id_to_idx' in encoders:
                # 역방향 조회
                for orig_item_id, orig_idx in encoders['item_id_to_idx'].items():
                    if orig_idx == idx:
                        item_id = orig_item_id
                        break
            
            if item_id:
                item_scores.append((item_id, similarity))
    
    # 상위 N개 선택
    item_scores.sort(key=lambda x: x[1], reverse=True)
    top_items = item_scores[:top_n]
    
    # 추천 결과 구성
    recommendations = []
    for item_id, score in top_items:
        # 아이템 메타데이터 가져오기
        item_data = all_user_data[all_user_data['item_id'] == item_id]
        if not item_data.empty:
            item_info = item_data.iloc[0]
            recommendations.append({
                'item_id': item_id,
                'title': item_info.get('title', 'Unknown'),
                'category': item_info.get('category', 'Unknown'),
                'price': int(item_info.get('price', 0)),
                'score': float(score),
                'reason': 'content_based'
            })
    
    return recommendations

@app.get("/")
def home():
    return {"message": "Splitty 추천 시스템 API"}

@app.get("/api/health")
def health_check():
    """시스템 상태 체크 엔드포인트"""
    return {
        "status": "healthy",
        "hybrid_model_loaded": hybrid_model_loaded,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.get("/api/stats")
def get_system_stats():
    """시스템 통계 정보 제공"""
    stats = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "hybrid_model_loaded": hybrid_model_loaded
    }
    
    # 하이브리드 모델 정보
    if hybrid_model_loaded and hybrid_recommender:
        try:
            hybrid_stats = hybrid_recommender.get_system_stats()
            stats["hybrid_model"] = hybrid_stats
        except Exception as e:
            stats["hybrid_model_error"] = str(e)
    
    return stats

@app.post("/api/recommend")
def recommend_items_with_filter(request: RecommendRequest):
    """
    통합 추천 API - available_items 필수, 페이징 지원
    
    요청 파라미터:
    - **user_id**: 추천을 받을 사용자의 ID (필수)
    - **top_n**: 반환할 상위 추천 아이템 개수 (기본값: 50, 최대: 100)
    - **categoryId**: 카테고리 ID 필터 (1-6, 옵션)
    - **available_items**: 거리 내 사용 가능한 아이템 ID 리스트 (필수)
    - **rank**: 페이징 시작 순위 (옵션, 기본값: 1)
    
    예시 요청:
    ```json
    {
        "user_id": "user_1",
        "top_n": 10,
        "categoryId": 1,
        "available_items": [101, 205, 310, 450, 892],
        "rank": 11
    }
    ```
    
    응답 형식:
    ```json
    {
        "user_id": "user_1",
        "items": [
            {"item_id": 297, "rank": 11},
            {"item_id": 418, "rank": 12}
        ]
    }
    ```
    """
    try:
        # available_items 필수 체크
        if not request.available_items:
            raise HTTPException(
                status_code=400,
                detail="available_items는 필수입니다."
            )
        
        # rank 기본값 설정 (1부터 시작)
        start_rank = request.rank if request.rank else 1
        
        # 1. 사용자 존재 여부 확인
        user_exists = check_user_exists(request.user_id)
        
        # 2. 추천 생성 (전체 추천 리스트)
        if user_exists and hybrid_model_loaded and hybrid_recommender:
            try:
                # 충분한 개수를 가져오기 위해 더 많이 요청
                # start_rank + top_n 만큼 필요
                fetch_count = start_rank + request.top_n - 1
                
                recommendations = hybrid_recommender.get_recommendations(
                    user_id=request.user_id,
                    top_k=250, 
                    top_n=min(fetch_count, 100),  # 최대 100개로 제한
                    category_filter=request.categoryId,
                    available_items=request.available_items
                )
                
            except Exception as hybrid_error:
                print(f"하이브리드 모델 오류: {str(hybrid_error)}")
                user_exists = False
        
        if not user_exists:
            # 신규 사용자 → 인기 아이템
            fetch_count = start_rank + request.top_n - 1
            recommendations = hybrid_recommender.get_popular_recommendations(
                top_n=min(fetch_count, 100),
                category_filter=request.categoryId,
                available_items=request.available_items
            ) if hybrid_model_loaded and hybrid_recommender else []
        
        # 3. 페이징 처리
        # start_rank는 1-based이므로 0-based 인덱스로 변환
        start_idx = start_rank - 1
        end_idx = start_idx + request.top_n
        
        # 슬라이싱
        paginated_recommendations = recommendations[start_idx:end_idx]
        
        # 4. rank 재조정 (start_rank부터 시작)
        simple_recommendations = []
        for i, rec in enumerate(paginated_recommendations):
            simple_recommendations.append({
                "item_id": rec.get("item_id"),
                "rank": start_rank + i
            })
        
        # 5. 응답 구성
        return {
            "user_id": request.user_id,
            "items": simple_recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"추천 처리 중 오류가 발생했습니다: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작시 데이터 로드"""
    try:
        # 하이브리드 추천 시스템 로드 시도
        load_hybrid_recommender()
        print("서버 시작 완료: 하이브리드 추천 시스템 준비됨")
    except Exception as e:
        print(f"경고: 하이브리드 모델 로드 실패 - {str(e)}")
        print("API 호출시 다시 시도됩니다.")

# 메인 실행부 (개발용)
if __name__ == "__main__":
    import uvicorn
    print("추천 시스템 서버를 시작합니다...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
