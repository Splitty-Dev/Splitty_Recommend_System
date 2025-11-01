"""
새 로그 데이터를 기존 학습 데이터와 병합하는 모듈
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import os
import json


class DataMerger:
    """기존 학습 데이터와 새 로그 데이터를 병합하는 클래스"""
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: 데이터 디렉토리 경로 (예: data/splitty_recommendation_data_1)
        """
        self.data_dir = data_dir
        self.train_file = os.path.join(data_dir, "user_item_train.csv")
        self.test_file = os.path.join(data_dir, "user_item_test.csv")
        self.val_file = os.path.join(data_dir, "user_item_val.csv")
        self.encoders_file = os.path.join(data_dir, "encoders.json")
    
    def load_existing_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        기존 학습 데이터를 로드합니다.
        
        Returns:
            (train_df, test_df, val_df)
        """
        print("기존 학습 데이터 로드 중...")
        
        train_df = pd.read_csv(self.train_file) if os.path.exists(self.train_file) else pd.DataFrame()
        test_df = pd.read_csv(self.test_file) if os.path.exists(self.test_file) else pd.DataFrame()
        val_df = pd.read_csv(self.val_file) if os.path.exists(self.val_file) else pd.DataFrame()
        
        print(f"Train: {len(train_df)}개, Test: {len(test_df)}개, Val: {len(val_df)}개")
        
        return train_df, test_df, val_df
    
    def merge_new_data(
        self,
        existing_train: pd.DataFrame,
        new_data: pd.DataFrame,
        deduplicate: bool = True,
        merge_strategy: str = "train_only"
    ) -> pd.DataFrame:
        """
        새 데이터를 기존 train 데이터와 병합합니다.
        
        Args:
            existing_train: 기존 train 데이터
            new_data: 새로 추가할 데이터
            deduplicate: 중복 제거 여부
            merge_strategy: 병합 전략
                - "train_only": train에만 추가 (권장, test/val 유지)
                - "full_resplit": 전체 재분할
            
        Returns:
            병합된 train 데이터
        """
        print(f"\n데이터 병합 시작 (전략: {merge_strategy})...")
        print(f"  기존 train: {len(existing_train)}개")
        print(f"  새 데이터: {len(new_data)}개")
        
        # 컬럼 체크
        required_columns = ['user_idx', 'item_idx', 'label', 'weight', 'price_norm', 'category_idx']
        for col in required_columns:
            if col not in new_data.columns:
                raise ValueError(f"새 데이터에 필수 컬럼이 없습니다: {col}")
        
        # 데이터 병합
        merged = pd.concat([existing_train, new_data], ignore_index=True)
        
        print(f"  병합 후: {len(merged)}개")
        
        # 중복 제거
        if deduplicate:
            print("  중복 제거 중...")
            # 같은 user-item 쌍이 있으면 더 높은 weight 유지
            merged = merged.sort_values('weight', ascending=False)
            before_dedup = len(merged)
            merged = merged.drop_duplicates(
                subset=['user_idx', 'item_idx'],
                keep='first'
            )
            print(f"  중복 제거: {before_dedup - len(merged)}개 제거됨")
        
        print(f"최종 데이터: {len(merged)}개")
        
        return merged
    
    def split_data(
        self,
        merged_data: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        병합된 데이터를 train/val/test로 분할합니다.
        
        Args:
            merged_data: 병합된 전체 데이터
            train_ratio: Train 비율
            val_ratio: Validation 비율
            test_ratio: Test 비율
            
        Returns:
            (train_df, val_df, test_df)
        """
        print(f"\n데이터 분할 중... (train: {train_ratio}, val: {val_ratio}, test: {test_ratio})")
        
        # 데이터 섞기
        shuffled = merged_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 분할 인덱스 계산
        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_df = shuffled[:train_end]
        val_df = shuffled[train_end:val_end]
        test_df = shuffled[val_end:]
        
        print(f"분할 완료: Train {len(train_df)}개, Val {len(val_df)}개, Test {len(test_df)}개")
        
        return train_df, val_df, test_df
    
    def save_merged_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        backup: bool = True
    ):
        """
        병합된 데이터를 저장합니다.
        
        Args:
            train_df: Train 데이터
            val_df: Validation 데이터
            test_df: Test 데이터
            backup: 기존 파일을 백업할지 여부
        """
        print("\n데이터 저장 중...")
        
        # 백업
        if backup:
            self._backup_existing_files()
        
        # 저장
        train_df.to_csv(self.train_file, index=False)
        val_df.to_csv(self.val_file, index=False)
        test_df.to_csv(self.test_file, index=False)
        
        print(f"저장 완료:")
        print(f"  Train: {self.train_file}")
        print(f"  Val: {self.val_file}")
        print(f"  Test: {self.test_file}")
    
    def _backup_existing_files(self):
        """
        기존 파일을 백업합니다.
        """
        from datetime import datetime
        
        backup_suffix = datetime.now().strftime("_%Y%m%d_%H%M%S.backup")
        
        for file_path in [self.train_file, self.val_file, self.test_file]:
            if os.path.exists(file_path):
                backup_path = file_path + backup_suffix
                os.rename(file_path, backup_path)
                print(f"  백업: {backup_path}")
    
    def get_data_stats(self, df: pd.DataFrame, name: str = "데이터") -> Dict:
        """
        데이터 통계를 반환합니다.
        
        Args:
            df: 데이터프레임
            name: 데이터 이름
            
        Returns:
            통계 딕셔너리
        """
        stats = {
            "name": name,
            "total_records": len(df),
            "unique_users": df['user_idx'].nunique(),
            "unique_items": df['item_idx'].nunique(),
            "positive_samples": (df['label'] == 1).sum() if 'label' in df.columns else 0,
            "negative_samples": (df['label'] == 0).sum() if 'label' in df.columns else 0,
            "avg_weight": df['weight'].mean() if 'weight' in df.columns else 0,
        }
        
        return stats
    
    def print_comparison(
        self,
        before_train: pd.DataFrame,
        after_train: pd.DataFrame
    ):
        """
        병합 전후 비교를 출력합니다.
        
        Args:
            before_train: 병합 전 train 데이터
            after_train: 병합 후 train 데이터
        """
        print("\n" + "="*60)
        print("데이터 병합 전후 비교")
        print("="*60)
        
        before_stats = self.get_data_stats(before_train, "병합 전")
        after_stats = self.get_data_stats(after_train, "병합 후")
        
        for key in before_stats:
            if key == "name":
                continue
            before_val = before_stats[key]
            after_val = after_stats[key]
            change = after_val - before_val if isinstance(before_val, (int, float)) else 0
            print(f"{key:20s}: {before_val:8} → {after_val:8} (+{change})")
        
        print("="*60)


# 사용 예시
if __name__ == "__main__":
    # 데이터 병합기 초기화
    data_dir = "../data/splitty_recommendation_data_1"
    merger = DataMerger(data_dir)
    
    # 기존 데이터 로드
    train_df, test_df, val_df = merger.load_existing_data()
    
    # 샘플 새 데이터 (실제로는 log_data_transformer에서 생성)
    new_data = pd.DataFrame({
        'user_idx': ['201', '202', '203'],
        'item_idx': [10, 20, 30],
        'label': [1, 1, 1],
        'weight': [2.0, 3.0, 1.0],
        'price_norm': [0.5, 0.7, 0.3],
        'category_idx': [1, 2, 3]
    })
    
    # 데이터 병합
    merged_train = merger.merge_new_data(train_df, new_data)
    
    # 비교 출력
    merger.print_comparison(train_df, merged_train)
    
    # 재분할
    new_train, new_val, new_test = merger.split_data(merged_train)
    
    # 저장 (실제 저장은 주석 처리)
    # merger.save_merged_data(new_train, new_val, new_test, backup=True)
