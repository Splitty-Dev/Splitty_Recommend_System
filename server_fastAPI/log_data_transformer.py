"""
S3 로그를 학습 데이터 포맷으로 변환하는 모듈
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime


class LogDataTransformer:
    """로그 데이터를 학습 데이터로 변환하는 클래스"""
    
    def __init__(self):
        """
        액션 타입별 가중치 정의
        """
        self.action_weights = {
            'VIEW': 1.0,      # 조회
            'CLICK': 2.0,     # 클릭
            'ADD_CART': 3.0,  # 장바구니 추가
            'PURCHASE': 5.0   # 구매 (가장 중요)
        }
    
    def transform_logs_to_training_data(
        self,
        logs: List[Dict],
        existing_max_user_id: int = 200,
        existing_max_item_id: int = 500
    ) -> pd.DataFrame:
        """
        S3 로그를 학습 데이터 포맷으로 변환합니다.
        
        로그 포맷:
        {"timestamp": 1762003990140, "item_id": 31, "user_id": 1, "action": "VIEW", "category_id": 1, "price": 20000}
        
        학습 데이터 포맷:
        user_idx, item_idx, label, weight, price_norm, category_idx
        
        Args:
            logs: S3에서 가져온 로그 리스트
            existing_max_user_id: 기존 데이터의 최대 user_id
            existing_max_item_id: 기존 데이터의 최대 item_id
            
        Returns:
            학습 데이터 DataFrame
        """
        if not logs:
            print("변환할 로그가 없습니다.")
            return pd.DataFrame()
        
        print(f"로그 변환 시작: {len(logs)}개 레코드")
        
        # 로그를 DataFrame으로 변환
        df = pd.DataFrame(logs)
        
        # 필수 컬럼 체크
        required_columns = ['timestamp', 'user_id', 'item_id', 'action', 'category_id', 'price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
        
        print(f"원본 로그: {len(df)}개 레코드")
        print(f"액션 분포:\n{df['action'].value_counts()}")
        
        # 1. user_idx 매핑 (기존 ID와 겹치지 않게)
        df['user_idx'] = df['user_id'].astype(str)
        
        # 2. item_idx 매핑 (기존 ID와 겹치지 않게)
        df['item_idx'] = df['item_id'].astype(int)
        
        # 3. label 생성 (모든 상호작용은 positive)
        df['label'] = 1
        
        # 4. weight 계산 (액션 타입별 가중치)
        df['weight'] = df['action'].map(self.action_weights).fillna(1.0)
        
        # 5. price_norm 계산 (가격 정규화: 0-1 범위)
        df['price_norm'] = self._normalize_price(df['price'])
        
        # 6. category_idx (1-6 범위로 제한)
        df['category_idx'] = df['category_id'].clip(1, 6).astype(int)
        
        # 7. 필요한 컬럼만 선택
        training_data = df[[
            'user_idx', 'item_idx', 'label', 
            'weight', 'price_norm', 'category_idx'
        ]].copy()
        
        # 8. 중복 제거 (같은 user-item 쌍은 가장 높은 weight만 유지)
        training_data = training_data.sort_values('weight', ascending=False)
        training_data = training_data.drop_duplicates(
            subset=['user_idx', 'item_idx'],
            keep='first'
        )
        
        print(f"변환 완료: {len(training_data)}개 학습 레코드")
        print(f"유니크 사용자: {training_data['user_idx'].nunique()}명")
        print(f"유니크 아이템: {training_data['item_idx'].nunique()}개")
        
        return training_data
    
    def _normalize_price(self, prices: pd.Series) -> pd.Series:
        """
        가격을 0-1 범위로 정규화합니다.
        
        Args:
            prices: 가격 시리즈
            
        Returns:
            정규화된 가격 시리즈
        """
        min_price = prices.min()
        max_price = prices.max()
        
        if max_price == min_price:
            return pd.Series([0.5] * len(prices))
        
        normalized = (prices - min_price) / (max_price - min_price)
        return normalized
    
    def create_negative_samples(
        self,
        positive_data: pd.DataFrame,
        negative_ratio: float = 0.2
    ) -> pd.DataFrame:
        """
        Negative 샘플을 생성합니다 (사용자가 상호작용하지 않은 아이템).
        
        Args:
            positive_data: Positive 학습 데이터
            negative_ratio: Negative 샘플 비율 (positive 대비)
            
        Returns:
            Negative 샘플 DataFrame
        """
        print(f"Negative 샘플 생성 시작 (비율: {negative_ratio})")
        
        # 모든 user-item 조합 생성
        all_users = positive_data['user_idx'].unique()
        all_items = positive_data['item_idx'].unique()
        
        # Positive 샘플 집합
        positive_pairs = set(
            zip(positive_data['user_idx'], positive_data['item_idx'])
        )
        
        # Negative 샘플 개수 계산
        n_negative = int(len(positive_data) * negative_ratio)
        
        negative_samples = []
        attempts = 0
        max_attempts = n_negative * 10  # 무한 루프 방지
        
        while len(negative_samples) < n_negative and attempts < max_attempts:
            # 랜덤으로 user-item 쌍 생성
            user = np.random.choice(all_users)
            item = np.random.choice(all_items)
            
            # Positive에 없는 쌍만 추가
            if (user, item) not in positive_pairs:
                # 해당 아이템의 평균 속성 사용
                item_data = positive_data[positive_data['item_idx'] == item]
                if len(item_data) > 0:
                    avg_price_norm = item_data['price_norm'].mean()
                    category = item_data['category_idx'].mode()[0] if len(item_data['category_idx'].mode()) > 0 else 1
                else:
                    avg_price_norm = 0.5
                    category = 1
                
                negative_samples.append({
                    'user_idx': user,
                    'item_idx': item,
                    'label': 0,  # Negative
                    'weight': 1.0,
                    'price_norm': avg_price_norm,
                    'category_idx': category
                })
            
            attempts += 1
        
        negative_df = pd.DataFrame(negative_samples)
        print(f"Negative 샘플 생성 완료: {len(negative_df)}개")
        
        return negative_df
    
    def get_transformation_stats(self, df: pd.DataFrame) -> Dict:
        """
        변환된 데이터의 통계를 반환합니다.
        
        Args:
            df: 변환된 학습 데이터
            
        Returns:
            통계 딕셔너리
        """
        return {
            "total_records": len(df),
            "unique_users": df['user_idx'].nunique(),
            "unique_items": df['item_idx'].nunique(),
            "positive_samples": (df['label'] == 1).sum(),
            "negative_samples": (df['label'] == 0).sum(),
            "avg_weight": df['weight'].mean(),
            "category_distribution": df['category_idx'].value_counts().to_dict()
        }


# 사용 예시
if __name__ == "__main__":
    # 샘플 로그 데이터
    sample_logs = [
        {"timestamp": 1762003990140, "item_id": 31, "user_id": "1", "action": "VIEW", "category_id": 1, "price": 20000},
        {"timestamp": 1762003995853, "item_id": 31, "user_id": "1", "action": "PURCHASE", "category_id": 1, "price": 20000},
        {"timestamp": 1762004000000, "item_id": 45, "user_id": "2", "action": "CLICK", "category_id": 3, "price": 10000},
    ]
    
    # 변환기 초기화
    transformer = LogDataTransformer()
    
    # 로그를 학습 데이터로 변환
    training_data = transformer.transform_logs_to_training_data(sample_logs)
    
    print("\n변환된 학습 데이터:")
    print(training_data)
    
    # 통계 출력
    stats = transformer.get_transformation_stats(training_data)
    print("\n데이터 통계:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
