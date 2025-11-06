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
        
        # 학습 실행
        self.two_tower_trainer.fit(
            train_df=self.train_data,
            epochs=epochs,
            batch_size=batch_size
        )
        
        self.is_trained = True
        print("\n=== 하이브리드 추천 시스템 학습 완료 ===")
    
    def get_recommendations(self, user_id: str, top_k: int = 250, top_n: int = 50, 
                           category_filter: int = None, available_items: List[int] = None) -> List[Dict]:
        """
        하이브리드 추천 실행
        
        Args:
            user_id: 사용자 ID
            top_k: Matrix Factorization에서 추출할 후보 개수
            top_n: 최종 반환할 추천 개수
            category_filter: 카테고리 ID (1-6: 1=식품, 2=생활/주방, 3=뷰티/미용, 4=패션, 5=건강/운동, 6=유아)
            available_items: 거리 내 사용 가능한 아이템 ID 리스트 (옵션) - 처음부터 이 범위 내에서만 추천
            
        Returns:
            추천 결과 리스트
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # category_filter는 이미 정수 (1-6)이므로 그대로 사용
        category_idx_filter = category_filter
        
        filter_msg = f" (카테고리 ID: {category_filter})" if category_filter else ""
        available_msg = f" (available_items: {len(available_items)} 개)" if available_items else ""
        print(f"\n사용자 {user_id}에 대한 하이브리드 추천 시작 (K={top_k}, N={top_n}){filter_msg}{available_msg}")
        
        # user_id를 정수로 변환 (encoder의 키가 int/numpy.int64이므로)
        try:
            user_id_int = int(user_id)
        except (ValueError, TypeError):
            user_id_int = user_id
        
        # 1단계: Matrix Factorization으로 Top-K 후보 생성 (available_items 범위 내에서)
        print("1단계: Matrix Factorization으로 후보 생성...")
        mf_candidates = self.matrix_factorization.get_top_k_candidates(
            user_id=user_id_int,  # 정수로 변환된 user_id 사용
            k=top_k,
            exclude_purchased=True,  # 구매한 아이템(weight=5)만 제외
            interaction_df=self.train_data,
            category_filter=category_idx_filter,
            available_items=available_items  # 처음부터 available_items로 제한
        )
        
        if not mf_candidates:
            print("Matrix Factorization에서 후보를 찾을 수 없습니다.")
            return []
        
        print(f"Matrix Factorization 후보 수: {len(mf_candidates)}")
        
        # 2단계: Two-Tower 모델로 개인화된 순위 매기기 (MF 후보들에 대해서만)
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
            user_id=user_id_int,  # 정수로 변환된 user_id 사용
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
        
        # Top-N까지만 처리 (이미 available_items로 필터링되어 있음)
        for i, (item_idx, tt_score) in enumerate(scored_items[:top_n]):
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
                'item_id': int(item_idx),  # numpy.int64 -> Python int 변환
                'score': float(final_score)
            }
            
            recommendations.append(recommendation)
        
        print(f"최종 추천 완료: {len(recommendations)} 개 아이템")
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
    
    def get_popular_recommendations(self, top_n: int = 50, category_filter: int = None, available_items: List[int] = None) -> List[Dict]:
        """인기 아이템 기반 추천 (새 사용자용)
        
        Args:
            top_n: 반환할 추천 개수
            category_filter: 카테고리 ID (1-6: 1=식품, 2=생활/주방, 3=뷰티/미용, 4=패션, 5=건강/운동, 6=유아)
            available_items: 거리 내 사용 가능한 아이템 ID 리스트
        """
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
        
        # available_items 필터링
        if available_items:
            popularity = popularity[popularity['item_idx'].isin(available_items)]
        
        # category 필터링 (category_filter는 1-6 정수)
        if category_filter and self.item_meta is not None:
            if 'category_idx' in self.item_meta.columns:
                filtered_items = self.item_meta[self.item_meta['category_idx'] == category_filter]['item_idx']
                popularity = popularity[popularity['item_idx'].isin(filtered_items)]
        
        popular_items = popularity.nlargest(top_n, 'popularity_score')
        
        recommendations = []
        for i, row in popular_items.iterrows():
            item_id = row['item_idx']
            
            # 아이템 메타데이터 가져오기
            item_info = self.item_meta[self.item_meta['item_idx'] == item_id]
            if item_info.empty:
                continue
            
            item_data = item_info.iloc[0]
            
            recommendation = {
                'item_id': int(item_id),  # numpy.int64 -> Python int 변환
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
            model_data = torch.load(tt_path, map_location=self.device, weights_only=False)
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