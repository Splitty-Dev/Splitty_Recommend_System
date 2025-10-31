"""
Hybrid Recommendation System
Matrix Factorization + Two-Tower Model을 결합한 하이브리드 추천 시스템

구조:
1. Matrix Factorization → Top-K Candidates (K=250)  
2. Two-Tower Model → Top-N Personalized Ranking (N=50)
3. Final Output → 사용자에게 추천
"""

import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Dict, Optional
from models import MatrixFactorization, ImplicitMatrixFactorization, TwoTowerModel, TwoTowerTrainer
import torch


class HybridRecommendationSystem:
    """
    Matrix Factorization과 Two-Tower 모델을 결합한 하이브리드 추천 시스템
    """
    
    # 카테고리 매핑 (문자열 -> 인덱스)
CATEGORY_MAPPING = {
    "건강/운동": 0,
    "뷰티/미용": 1, 
    "생활/주방": 2,
    "식품": 3,
    "유아": 4
}

# 역매핑 (인덱스 -> 카테고리명)
CATEGORY_NAMES = {v: k for k, v in CATEGORY_MAPPING.items()}

class HybridRecommender:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.matrix_factorization = None
        self.two_tower_trainer = None
        self.item_meta = None
        self.train_data = None
        self.is_trained = False
        
    def load_data(self, data_dir: str):
        """데이터 로드"""
        print("데이터 로딩 시작...")
        
        # 학습 데이터 로드
        train_path = os.path.join(data_dir, "user_item_train.csv")
        if os.path.exists(train_path):
            self.train_data = pd.read_csv(train_path)
            print(f"학습 데이터 로드 완료: {self.train_data.shape}")
        else:
            raise FileNotFoundError(f"학습 데이터를 찾을 수 없습니다: {train_path}")
        
        # 아이템 메타데이터 준비 (학습 데이터에서 추출)
        available_cols = self.train_data.columns.tolist()
        print(f"사용 가능한 컬럼: {available_cols}")
        
        # 실제 데이터 구조에 맞춘 컬럼 선택
        meta_cols = ['item_idx', 'category_idx']
        if 'price_norm' in available_cols:
            meta_cols.append('price_norm')
        if 'title_emb' in available_cols:
            meta_cols.append('title_emb')
        
        self.item_meta = self.train_data[meta_cols].drop_duplicates()
        print(f"아이템 메타데이터 준비 완료: {len(self.item_meta)} 개 아이템")
        
        print("데이터 로딩 완료!")
        
    def train_models(self, mf_factors: int = 50, epochs: int = 50, batch_size: int = 1024):
        """하이브리드 모델 학습"""
        if self.train_data is None:
            raise ValueError("데이터를 먼저 로드해야 합니다.")
        
        print("=== 하이브리드 추천 시스템 학습 시작 ===")
        
        # 1단계: Implicit Matrix Factorization 학습 (네거티브 샘플링 포함)
        print("\n1단계: Implicit Matrix Factorization 학습 (네거티브 샘플링 포함)")
        self.matrix_factorization = ImplicitMatrixFactorization(
            n_factors=mf_factors,
            learning_rate=0.01,
            regularization=0.01,
            n_epochs=100,
            negative_samples=5  # 1:5 비율로 네거티브 샘플링
        )
        self.matrix_factorization.fit(self.train_data, use_negative_sampling=True)
        
        # 2단계: Two-Tower 모델 준비 및 학습
        print("\n2단계: Two-Tower 모델 학습")
        
        # 고유값 개수 계산
        n_users = self.train_data['user_idx'].nunique()
        n_items = self.train_data['item_idx'].nunique() 
        n_categories = self.train_data['category_idx'].nunique()
        
        print(f"사용자 수: {n_users}, 아이템 수: {n_items}, 카테고리 수: {n_categories}")
        
        # Two-Tower 모델 초기화
        two_tower_model = TwoTowerModel(
            n_users=n_users,
            n_items=n_items,
            n_categories=n_categories,
            embedding_dim=64,
            hidden_dims=[128, 64]
        )
        
        self.two_tower_trainer = TwoTowerTrainer(two_tower_model, device=self.device)
        
        # 학습 실행 (제목 임베딩 포함)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        data_path = os.path.join(project_root, "data", "splitty_recommendation_data_1")
        
        self.two_tower_trainer.fit(
            train_df=self.train_data,
            epochs=epochs,
            batch_size=batch_size,
            data_dir=data_path
        )
        
        self.is_trained = True
        print("\n=== 하이브리드 추천 시스템 학습 완료 ===")
    
    def get_recommendations(self, user_id: str, top_k: int = 250, top_n: int = 50, 
                           category_filter: str = None, available_items: List[int] = None) -> List[Dict]:
        """
        하이브리드 추천 실행
        
        Args:
            user_id: 사용자 ID
            top_k: Matrix Factorization에서 추출할 후보 개수
            top_n: 최종 반환할 추천 개수
            category_filter: 특정 카테고리로 필터링 (옵션)
            available_items: 거리 내 사용 가능한 아이템 ID 리스트 (옵션)
            
        Returns:
            추천 결과 리스트
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # 카테고리 필터링을 위해 문자열을 인덱스로 변환
        category_idx_filter = None
        if category_filter:
            category_idx_filter = self.CATEGORY_MAPPING.get(category_filter)
            if category_idx_filter is None:
                print(f"경고: 알 수 없는 카테고리 '{category_filter}'. 가능한 카테고리: {list(self.CATEGORY_MAPPING.keys())}")
                return []
        
        filter_msg = f" (카테고리: {category_filter})" if category_filter else ""
        print(f"\n사용자 {user_id}에 대한 하이브리드 추천 시작 (K={top_k}, N={top_n}){filter_msg}")
        
        # 1단계: Matrix Factorization으로 Top-K 후보 생성
        print("1단계: Matrix Factorization으로 후보 생성...")
        mf_candidates = self.matrix_factorization.get_top_k_candidates(
            user_id=user_id,
            k=top_k,
            exclude_seen=True,
            interaction_df=self.train_data,
            category_filter=category_idx_filter,
            available_items=available_items
        )
        
        if not mf_candidates:
            print("Matrix Factorization에서 후보를 찾을 수 없습니다.")
            return []
        
        print(f"Matrix Factorization 후보 수: {len(mf_candidates)}")
        
        # 2단계: Two-Tower 모델로 개인화된 순위 매기기
        print("2단계: Two-Tower 모델로 순위 매기기...")
        candidate_items = [item_idx for item_idx, _ in mf_candidates]
        
        # 아이템 메타데이터에 price_norm 추가 (없는 경우)
        item_meta_with_norm = self._prepare_item_meta_for_prediction()
        
        # 카테고리 필터링이 있는 경우 메타데이터도 필터링
        if category_idx_filter is not None:
            item_meta_with_norm = item_meta_with_norm[
                item_meta_with_norm['category_idx'] == category_idx_filter
            ]
        
        scored_items = self.two_tower_trainer.predict_scores(
            user_id=user_id,
            candidate_items=candidate_items,
            item_meta=item_meta_with_norm
        )
        
        if not scored_items:
            print("Two-Tower 모델에서 점수를 계산할 수 없습니다.")
            return []
        
        print(f"Two-Tower 점수 계산 완료: {len(scored_items)} 개 아이템")
        
        # 3단계: 최종 추천 결과 구성
        print("3단계: 최종 추천 결과 구성...")
        recommendations = []
        
        # available_items 필터링을 위한 set 변환
        available_items_set = set(available_items) if available_items else None
        
        for i, (item_idx, tt_score) in enumerate(scored_items[:top_n * 2]):  # 필터링을 고려해 더 많이 가져옴
            # available_items 필터링 적용
            if available_items_set and item_idx not in available_items_set:
                continue
                
            # MF 점수 찾기
            mf_score = next((score for iid, score in mf_candidates if iid == item_idx), 0.0)
            
            # 아이템 메타데이터 가져오기
            item_info = self.item_meta[self.item_meta['item_idx'] == item_idx]
            if item_info.empty:
                continue
            
            item_data = item_info.iloc[0]
            
            # 최종 점수 (가중 평균)
            final_score = 0.6 * tt_score + 0.4 * (mf_score / 10.0 if mf_score > 0 else 0)
            
            recommendation = {
                'item_id': item_idx,
                'score': float(final_score)
            }
            
            recommendations.append(recommendation)
            
            # 충분한 개수가 모이면 중단
            if len(recommendations) >= top_n:
                break
        
        filter_msg = f" (available_items: {len(available_items_set)} 개)" if available_items_set else ""
        print(f"최종 추천 완료: {len(recommendations)} 개 아이템{filter_msg}")
        return recommendations
    
    def _prepare_item_meta_for_prediction(self) -> pd.DataFrame:
        """예측용 아이템 메타데이터 준비"""
        if 'price_norm' not in self.item_meta.columns:
            # price_norm이 없는 경우 정규화 수행
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            price_values = self.item_meta[['price']].values
            price_norm = scaler.fit_transform(price_values).flatten()
            
            item_meta_copy = self.item_meta.copy()
            item_meta_copy['price_norm'] = price_norm
            return item_meta_copy
        
        return self.item_meta
    
    def get_popular_recommendations(self, top_n: int = 50) -> List[Dict]:
        """인기 아이템 기반 추천 (새 사용자용)"""
        if self.train_data is None:
            return []
        
        # 가중치 기반 인기도 계산
        popularity = self.train_data.groupby('item_idx').agg({
            'weight': 'sum',
            'user_idx': 'nunique'  # 상호작용한 사용자 수
        }).reset_index()
        
        # 복합 점수 (가중치 합 + 사용자 다양성)
        popularity['popularity_score'] = (
            0.7 * (popularity['weight'] / popularity['weight'].max()) +
            0.3 * (popularity['user_idx'] / popularity['user_idx'].max())
        )
        
        popular_items = popularity.nlargest(top_n, 'popularity_score')
        
        recommendations = []
        for i, row in popular_items.iterrows():
            item_id = row['item_id']
            
            # 아이템 메타데이터 가져오기
            item_info = self.item_meta[self.item_meta['item_id'] == item_id]
            if item_info.empty:
                continue
            
            item_data = item_info.iloc[0]
            
            recommendation = {
                'item_id': item_id,
                'title': item_data.get('title', 'Unknown'),
                'category': item_data.get('category', 'Unknown'),
                'price': int(item_data.get('price', 0)),
                'final_score': float(row['popularity_score']),
                'two_tower_score': 0.0,
                'mf_score': 0.0,
                'rank': len(recommendations) + 1,
                'reason': 'popular_item'
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def save_models(self, model_dir: str):
        """모델들 저장"""
        os.makedirs(model_dir, exist_ok=True)
        
        if self.matrix_factorization:
            mf_path = os.path.join(model_dir, "matrix_factorization.pkl")
            self.matrix_factorization.save_model(mf_path)
        
        if self.two_tower_trainer:
            tt_path = os.path.join(model_dir, "two_tower_model.pth")
            self.two_tower_trainer.save_model(tt_path)
        
        print(f"모델들이 {model_dir}에 저장되었습니다.")
    
    def load_models(self, model_dir: str, data_dir: str):
        """모델들 로드"""
        print("모델 로딩 시작...")
        
        # 데이터 먼저 로드
        self.load_data(data_dir)
        
        # Implicit Matrix Factorization 로드
        mf_path = os.path.join(model_dir, "matrix_factorization.pkl")
        if os.path.exists(mf_path):
            self.matrix_factorization = ImplicitMatrixFactorization()
            self.matrix_factorization.load_model(mf_path)
        
        # Two-Tower 모델 로드  
        tt_path = os.path.join(model_dir, "two_tower_model.pth")
        if os.path.exists(tt_path):
            # 모델 설정 정보 로드
            model_data = torch.load(tt_path, map_location=self.device)
            config = model_data['model_config']
            
            # 모델 재생성
            two_tower_model = TwoTowerModel(
                n_users=config['n_users'],
                n_items=config['n_items'],
                n_categories=config['n_categories'],
                embedding_dim=config['embedding_dim'],
                hidden_dims=config['hidden_dims']
            )
            
            self.two_tower_trainer = TwoTowerTrainer(two_tower_model, device=self.device)
            self.two_tower_trainer.load_model(tt_path)
        
        self.is_trained = True
        print("모델 로딩 완료!")
    
    def get_system_stats(self) -> Dict:
        """시스템 통계 정보"""
        if not self.is_trained or self.train_data is None:
            return {"error": "모델이 학습되지 않았거나 데이터가 없습니다."}
        
        stats = {
            "is_trained": self.is_trained,
            "train_data_size": len(self.train_data),
            "unique_users": self.train_data['user_idx'].nunique(),
            "unique_items": self.train_data['item_idx'].nunique(),
            "unique_categories": self.train_data['category_idx'].nunique(),
            "total_interactions": len(self.train_data),
            "device": self.device
        }
        
        if self.matrix_factorization:
            stats["mf_factors"] = self.matrix_factorization.n_factors
            stats["mf_users"] = len(self.matrix_factorization.user_encoder)
            stats["mf_items"] = len(self.matrix_factorization.item_encoder)
        
        return stats
    
    def _get_category_name(self, category_idx: int) -> str:
        """카테고리 인덱스를 카테고리 명으로 변환"""
        return CATEGORY_NAMES.get(category_idx, "Unknown")