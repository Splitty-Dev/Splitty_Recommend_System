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
    category: Optional[str] = Field(None, description="특정 카테고리로 필터링")
    available_items: Optional[List[int]] = Field(None, description="거리 내 사용 가능한 아이템 ID 리스트")

class LocationBasedRequest(BaseModel):
    user_id: str = Field(..., description="사용자 ID")
    top_n: int = Field(50, ge=1, le=100, description="상위 추천 개수")
    category: Optional[str] = Field(None, description="특정 카테고리로 필터링")
    location: Optional[Dict] = Field(None, description="위치 정보 (lat, lng, radius)")
    region_id: Optional[str] = Field(None, description="지역 ID")

# 글로벌 변수로 데이터 저장
item_embeddings = None
encoders = None
user_item_test = None
user_item_val = None
data_loaded = False

# 하이브리드 추천 시스템
hybrid_recommender = None
hybrid_model_loaded = False

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

@app.get("/api/recommend")
def recommend_items(
    user_id: str = Query(..., description="사용자 ID"),
    top_n: int = Query(50, ge=1, le=100, description="상위 추천 개수 (1-100)"),
    category: Optional[str] = Query(None, description="특정 카테고리로 필터링 (옵션)")
):
    """
    사용자에게 아이템을 추천하는 API 엔드포인트 - 아이템 ID와 순위만 반환
    
    - **user_id**: 추천을 받을 사용자의 ID
    - **top_n**: 반환할 상위 추천 아이템 개수 (기본값: 50, 최대: 100)
    - **category**: 특정 카테고리로 필터링 (옵션, 없으면 전체 카테고리)
    
    예시: 
    - 전체 추천: /api/recommend?user_id=user123&top_n=10
    - 카테고리 필터링: /api/recommend?user_id=user123&top_n=10&category=전자제품
    """
    try:
        # 하이브리드 모델 시도
        if hybrid_model_loaded and hybrid_recommender:
            try:
                recommendations = hybrid_recommender.get_recommendations(
                    user_id=user_id,
                    top_k=250, 
                    top_n=top_n,
                    category_filter=category
                )
                
                # 간단한 응답 형태로 변환
                simple_recommendations = [
                    {
                        "item_id": rec["item_id"],
                        "rank": rec["rank"]
                    }
                    for rec in recommendations
                ]
                
                return {
                    "user_id": user_id,
                    "items": simple_recommendations
                }
                
            except Exception as hybrid_error:
                print(f"하이브리드 모델 오류: {str(hybrid_error)}")
                # 기존 모델로 폴백
        
        # 기존 콘텐츠 기반 모델 사용
        if not data_loaded:
            load_recommendation_data()
        
        # 기존 추천 결과 생성
        recommendations = get_user_recommendations(user_id, top_n)
        
        # 간단한 응답 형태로 변환
        simple_recommendations = [
            {
                "item_id": rec.get("item_id", ""),
                "rank": i + 1
            }
            for i, rec in enumerate(recommendations)
        ]
        
        return {
            "user_id": user_id,
            "items": simple_recommendations
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"추천 처리 중 오류가 발생했습니다: {str(e)}"
        )

@app.get("/api/health")
def health_check():
    """시스템 상태 체크 엔드포인트"""
    return {
        "status": "healthy",
        "data_loaded": data_loaded,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.get("/api/stats")
def get_system_stats():
    """시스템 통계 정보 제공"""
    stats = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "hybrid_model_loaded": hybrid_model_loaded,
        "legacy_data_loaded": data_loaded
    }
    
    # 하이브리드 모델 정보
    if hybrid_model_loaded and hybrid_recommender:
        try:
            hybrid_stats = hybrid_recommender.get_system_stats()
            stats["hybrid_model"] = hybrid_stats
        except Exception as e:
            stats["hybrid_model_error"] = str(e)
    
    # 기존 시스템 정보 (폴백용)
    if data_loaded:
        legacy_stats = {}
        
        # 임베딩 정보
        if item_embeddings is not None:
            legacy_stats["embeddings"] = {
                "total_items": item_embeddings.shape[0],
                "embedding_dimension": item_embeddings.shape[1]
            }
        
        # 인코더 정보
        if encoders is not None:
            legacy_stats["encoders"] = {
                "available_encoders": list(encoders.keys())
            }
        
        # 데이터셋 정보
        if user_item_test is not None:
            legacy_stats["test_data"] = {
                "rows": len(user_item_test),
                "unique_users": user_item_test['user_id'].nunique() if 'user_id' in user_item_test.columns else 0,
                "unique_items": user_item_test['item_id'].nunique() if 'item_id' in user_item_test.columns else 0
            }
        
        if user_item_val is not None:
            legacy_stats["validation_data"] = {
                "rows": len(user_item_val),
                "unique_users": user_item_val['user_id'].nunique() if 'user_id' in user_item_val.columns else 0,
                "unique_items": user_item_val['item_id'].nunique() if 'item_id' in user_item_val.columns else 0
            }
        
        stats["legacy_model"] = legacy_stats
    
    return stats

@app.get("/api/popular")
def get_popular_items(
    top_n: int = Query(50, ge=1, le=100, description="인기 아이템 개수")
):
    """인기 아이템 추천 (신규 사용자용)"""
    try:
        if hybrid_model_loaded and hybrid_recommender:
            recommendations = hybrid_recommender.get_popular_recommendations(top_n=top_n)
            
            return {
                "requested_count": top_n,
                "actual_count": len(recommendations),
                "model_type": "hybrid_popular",
                "recommendations": recommendations,
                "timestamp": pd.Timestamp.now().isoformat()
            }
        
        # 폴백: 기존 방식
        if not data_loaded:
            load_recommendation_data()
        
        return {
            "error": "하이브리드 모델을 사용할 수 없습니다. 기존 방식은 인기 아이템 기능을 지원하지 않습니다.",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"인기 아이템 추천 중 오류가 발생했습니다: {str(e)}"
        )

@app.post("/api/recommend")
def recommend_items_with_filter(request: RecommendRequest):
    """
    거리 기반 필터링이 포함된 아이템 추천 API
    
    - **user_id**: 추천을 받을 사용자의 ID
    - **top_n**: 반환할 상위 추천 아이템 개수 (기본값: 50, 최대: 100)
    - **category**: 특정 카테고리로 필터링 (옵션)
    - **available_items**: 거리 내 사용 가능한 아이템 ID 리스트 (옵션)
    
    예시 요청:
    ```json
    {
        "user_id": "user123",
        "top_n": 10,
        "category": "건강/운동",
        "available_items": [101, 205, 310, 450, 892]
    }
    ```
    """
    try:
        # 하이브리드 모델 시도
        if hybrid_model_loaded and hybrid_recommender:
            try:
                recommendations = hybrid_recommender.get_recommendations(
                    user_id=request.user_id,
                    top_k=250, 
                    top_n=request.top_n,
                    category_filter=request.category,
                    available_items=request.available_items
                )
                
                # 간단한 응답 형태로 변환
                simple_recommendations = [
                    {
                        "item_id": rec["item_id"],
                        "rank": rec["rank"]
                    }
                    for rec in recommendations
                ]
                
                return {
                    "user_id": request.user_id,
                    "items": simple_recommendations
                }
                
            except Exception as hybrid_error:
                print(f"하이브리드 모델 오류: {str(hybrid_error)}")
                # 기존 모델로 폴백
        
        # 기존 모델 사용 (available_items 필터링 포함)
        if not data_loaded:
            load_recommendation_data()
        
        recommendations = get_user_recommendations(request.user_id, request.top_n)
        
        # available_items 필터링 적용
        if request.available_items:
            available_set = set(request.available_items)
            recommendations = [
                rec for rec in recommendations 
                if rec.get("item_id") in available_set
            ]
            recommendations = recommendations[:request.top_n]
        
        # 간단한 응답 형태로 변환
        simple_recommendations = [
            {
                "item_id": rec.get("item_id", ""),
                "rank": i + 1
            }
            for i, rec in enumerate(recommendations)
        ]
        
        return {
            "user_id": request.user_id,
            "items": simple_recommendations
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"추천 처리 중 오류가 발생했습니다: {str(e)}"
        )

@app.post("/api/recommend/location")
def recommend_items_by_location(request: LocationBasedRequest):
    """
    위치 기반 아이템 추천 API (향후 확장용)
    
    현재는 region_id 기반으로만 동작하며, 실제 위치 계산은 백엔드에서 처리 후
    available_items 리스트와 함께 /api/recommend POST API를 사용하는 것을 권장합니다.
    """
    try:
        if request.location:
            # 위치 기반 로직은 백엔드에서 처리 후 available_items로 전달하는 것을 권장
            return {
                "message": "위치 기반 필터링은 백엔드에서 처리 후 /api/recommend POST API를 사용해주세요",
                "recommendation": "available_items 파라미터를 사용하세요",
                "example": {
                    "user_id": request.user_id,
                    "category": request.category,
                    "available_items": "거리 내 아이템 ID 리스트"
                }
            }
        
        # region_id 기반 처리 (향후 구현)
        if request.region_id:
            return {
                "message": f"지역 ID '{request.region_id}' 기반 추천은 아직 구현되지 않았습니다",
                "recommendation": "available_items 파라미터를 사용해주세요"
            }
        
        # 일반 추천으로 폴백
        basic_request = RecommendRequest(
            user_id=request.user_id,
            top_n=request.top_n,
            category=request.category
        )
        return recommend_items_with_filter(basic_request)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"위치 기반 추천 처리 중 오류가 발생했습니다: {str(e)}"
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
        print("기존 콘텐츠 기반 모델로 폴백합니다.")
        try:
            load_recommendation_data()
            print("서버 시작 완료: 기존 추천 시스템 준비됨")
        except Exception as e2:
            print(f"경고: 모든 모델 로드 실패 - {str(e2)}")
            print("API 호출시 다시 시도됩니다.")

# 메인 실행부 (개발용)
if __name__ == "__main__":
    import uvicorn
    print("추천 시스템 서버를 시작합니다...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
