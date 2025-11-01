#!/usr/bin/env python3
"""
하이브리드 추천 시스템 하이퍼파라미터 튜닝 스크립트

Grid Search를 사용하여 최적의 하이퍼파라미터를 찾습니다.
Validation set으로 평가하여 과적합을 방지합니다.

평가 지표:
- Precision@K: 추천한 아이템 중 실제로 좋아한 비율
- Recall@K: 좋아할 아이템 중 추천한 비율
- NDCG@K: 순위를 고려한 추천 품질
- Hit Rate@K: 최소 1개라도 맞춘 사용자 비율

사용법:
python hyperparameter_tuning.py
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from itertools import product
import json
from datetime import datetime
from hybrid_recommender import HybridRecommender

class RecommendationEvaluator:
    """추천 시스템 평가 클래스"""
    
    @staticmethod
    def precision_at_k(recommended_items: List[int], relevant_items: List[int], k: int) -> float:
        """Precision@K 계산"""
        if k == 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        relevant_set = set(relevant_items)
        
        hits = sum(1 for item in recommended_k if item in relevant_set)
        return hits / k
    
    @staticmethod
    def recall_at_k(recommended_items: List[int], relevant_items: List[int], k: int) -> float:
        """Recall@K 계산"""
        if len(relevant_items) == 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        relevant_set = set(relevant_items)
        
        hits = sum(1 for item in recommended_k if item in relevant_set)
        return hits / len(relevant_items)
    
    @staticmethod
    def dcg_at_k(recommended_items: List[int], relevant_items: List[int], k: int) -> float:
        """DCG@K 계산"""
        recommended_k = recommended_items[:k]
        relevant_set = set(relevant_items)
        
        dcg = 0.0
        for i, item in enumerate(recommended_k):
            if item in relevant_set:
                # 관련도는 1 (binary relevance)
                dcg += 1.0 / np.log2(i + 2)  # i+2 because index starts at 0
        
        return dcg
    
    @staticmethod
    def ndcg_at_k(recommended_items: List[int], relevant_items: List[int], k: int) -> float:
        """NDCG@K 계산"""
        dcg = RecommendationEvaluator.dcg_at_k(recommended_items, relevant_items, k)
        
        # Ideal DCG (모든 관련 아이템이 상위에 있을 때)
        ideal_relevant = relevant_items[:k]
        idcg = RecommendationEvaluator.dcg_at_k(ideal_relevant, relevant_items, k)
        
        if idcg == 0.0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def hit_rate_at_k(recommended_items: List[int], relevant_items: List[int], k: int) -> float:
        """Hit Rate@K 계산 (최소 1개라도 맞추면 1, 아니면 0)"""
        recommended_k = recommended_items[:k]
        relevant_set = set(relevant_items)
        
        hits = sum(1 for item in recommended_k if item in relevant_set)
        return 1.0 if hits > 0 else 0.0
    
    @classmethod
    def evaluate_recommendations(cls, recommender: HybridRecommender, 
                                val_data: pd.DataFrame, 
                                top_k: int = 250,
                                top_n: int = 50,
                                eval_k_list: List[int] = [5, 10, 20]) -> Dict:
        """Validation set에 대한 추천 평가"""
        
        print(f"\nValidation set 평가 시작 (top_k={top_k}, top_n={top_n})...")
        
        # 사용자별 관련 아이템 (validation set에서 상호작용한 아이템)
        user_relevant_items = val_data.groupby('user_idx')['item_idx'].apply(list).to_dict()
        
        results = {k: {
            'precision': [],
            'recall': [],
            'ndcg': [],
            'hit_rate': []
        } for k in eval_k_list}
        
        total_users = len(user_relevant_items)
        evaluated_users = 0
        
        for i, (user_id, relevant_items) in enumerate(user_relevant_items.items()):
            if i % 10 == 0:
                print(f"  진행률: {i}/{total_users} 사용자 평가 완료...", end='\r')
            
            try:
                # 추천 생성
                recommendations = recommender.get_recommendations(
                    user_id=str(user_id),
                    top_k=top_k,
                    top_n=top_n
                )
                
                if not recommendations:
                    continue
                
                # 추천된 아이템 ID 리스트
                recommended_items = [rec['item_id'] for rec in recommendations]
                
                # 각 K에 대해 평가
                for k in eval_k_list:
                    results[k]['precision'].append(
                        cls.precision_at_k(recommended_items, relevant_items, k)
                    )
                    results[k]['recall'].append(
                        cls.recall_at_k(recommended_items, relevant_items, k)
                    )
                    results[k]['ndcg'].append(
                        cls.ndcg_at_k(recommended_items, relevant_items, k)
                    )
                    results[k]['hit_rate'].append(
                        cls.hit_rate_at_k(recommended_items, relevant_items, k)
                    )
                
                evaluated_users += 1
                
            except Exception as e:
                # 추천 실패한 사용자는 스킵
                continue
        
        print(f"\n  평가 완료: {evaluated_users}/{total_users} 사용자")
        
        # 평균 계산
        metrics = {}
        for k in eval_k_list:
            metrics[f'Precision@{k}'] = np.mean(results[k]['precision']) if results[k]['precision'] else 0.0
            metrics[f'Recall@{k}'] = np.mean(results[k]['recall']) if results[k]['recall'] else 0.0
            metrics[f'NDCG@{k}'] = np.mean(results[k]['ndcg']) if results[k]['ndcg'] else 0.0
            metrics[f'HitRate@{k}'] = np.mean(results[k]['hit_rate']) if results[k]['hit_rate'] else 0.0
        
        metrics['evaluated_users'] = evaluated_users
        metrics['total_users'] = total_users
        
        return metrics


def grid_search_hyperparameters(data_path: str, 
                                model_save_path: str,
                                param_grid: Dict,
                                eval_k_list: List[int] = [5, 10, 20],
                                device: str = 'cpu'):
    """Grid Search로 최적 하이퍼파라미터 찾기"""
    
    print("=" * 80)
    print("하이퍼파라미터 Grid Search 시작")
    print("=" * 80)
    
    # Validation 데이터 로드
    val_path = os.path.join(data_path, "user_item_val.csv")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation 데이터를 찾을 수 없습니다: {val_path}")
    
    val_data = pd.read_csv(val_path)
    print(f"\nValidation 데이터 로드 완료: {val_data.shape}")
    print(f"  - 사용자 수: {val_data['user_idx'].nunique()}")
    print(f"  - 아이템 수: {val_data['item_idx'].nunique()}")
    print(f"  - 상호작용 수: {len(val_data)}")
    
    # Grid Search 파라미터 조합 생성
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    total_combinations = len(param_combinations)
    print(f"\n총 {total_combinations}개의 파라미터 조합을 테스트합니다.")
    print(f"평가 지표: Precision, Recall, NDCG, Hit Rate @ K={eval_k_list}")
    
    # 결과 저장
    all_results = []
    best_score = -1
    best_params = None
    best_metrics = None
    
    # 각 파라미터 조합에 대해 학습 및 평가
    for idx, params in enumerate(param_combinations):
        param_dict = dict(zip(param_names, params))
        
        print("\n" + "=" * 80)
        print(f"실험 {idx + 1}/{total_combinations}")
        print("-" * 80)
        print("파라미터:")
        for key, value in param_dict.items():
            print(f"  {key}: {value}")
        print("-" * 80)
        
        try:
            # 추천 시스템 초기화
            recommender = HybridRecommender(device=device)
            
            # 데이터 로드
            print("\n1. 데이터 로드 중...")
            recommender.load_data(data_path)
            
            # 모델 학습
            print("\n2. 모델 학습 중...")
            recommender.train_models(
                mf_factors=param_dict.get('mf_factors', 50),
                epochs=param_dict.get('epochs', 30),
                batch_size=param_dict.get('batch_size', 512)
            )
            
            # Validation set 평가
            print("\n3. Validation set 평가 중...")
            metrics = RecommendationEvaluator.evaluate_recommendations(
                recommender=recommender,
                val_data=val_data,
                top_k=param_dict.get('top_k', 250),
                top_n=param_dict.get('top_n', 50),
                eval_k_list=eval_k_list
            )
            
            # 결과 출력
            print("\n평가 결과:")
            for metric_name, value in metrics.items():
                if metric_name not in ['evaluated_users', 'total_users']:
                    print(f"  {metric_name}: {value:.4f}")
            
            # 종합 점수 계산 (NDCG@10을 주요 지표로 사용)
            composite_score = metrics.get('NDCG@10', 0.0)
            
            # 결과 저장
            result = {
                'experiment_id': idx + 1,
                'params': param_dict,
                'metrics': metrics,
                'composite_score': composite_score
            }
            all_results.append(result)
            
            # 최고 점수 업데이트
            if composite_score > best_score:
                best_score = composite_score
                best_params = param_dict
                best_metrics = metrics
                
                print(f"\n🎉 새로운 최고 점수! NDCG@10: {best_score:.4f}")
                
                # 최고 성능 모델 저장
                print(f"\n최고 성능 모델 저장 중...")
                best_model_path = os.path.join(model_save_path, "best_model")
                recommender.save_models(best_model_path)
                
                # 최고 파라미터 저장
                best_params_file = os.path.join(model_save_path, "best_params.json")
                with open(best_params_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'params': best_params,
                        'metrics': best_metrics,
                        'composite_score': best_score
                    }, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"\n❌ 실험 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 최종 결과 요약
    print("\n" + "=" * 80)
    print("Grid Search 완료!")
    print("=" * 80)
    
    if best_params:
        print("\n🏆 최적 하이퍼파라미터:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        print(f"\n📊 최고 성능 지표 (NDCG@10: {best_score:.4f}):")
        for metric_name, value in best_metrics.items():
            if metric_name not in ['evaluated_users', 'total_users']:
                print(f"  {metric_name}: {value:.4f}")
    
    # 전체 결과를 CSV로 저장
    results_file = os.path.join(model_save_path, f"grid_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    results_df_data = []
    for result in all_results:
        row = result['params'].copy()
        row.update(result['metrics'])
        row['composite_score'] = result['composite_score']
        results_df_data.append(row)
    
    results_df = pd.DataFrame(results_df_data)
    results_df.to_csv(results_file, index=False, encoding='utf-8')
    print(f"\n전체 결과가 저장되었습니다: {results_file}")
    
    return best_params, best_metrics, all_results


def main():
    """메인 함수"""
    
    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data", "splitty_recommendation_data_1")
    model_save_path = os.path.join(current_dir, "saved_models")
    
    os.makedirs(model_save_path, exist_ok=True)
    
    print(f"데이터 경로: {data_path}")
    print(f"모델 저장 경로: {model_save_path}")
    
    # Grid Search 파라미터 정의
    param_grid = {
        'mf_factors': [30, 50, 70],           # Matrix Factorization 잠재 요인 수
        'epochs': [20, 30, 40],                # Two-Tower 학습 에포크
        'batch_size': [256, 512, 1024],       # 배치 크기
        'top_k': [150, 250, 350],             # MF 후보 개수
        'top_n': [50],                         # 최종 추천 개수 (고정)
    }
    
    # 평가할 K 값들
    eval_k_list = [5, 10, 20]
    
    # Grid Search 실행
    best_params, best_metrics, all_results = grid_search_hyperparameters(
        data_path=data_path,
        model_save_path=model_save_path,
        param_grid=param_grid,
        eval_k_list=eval_k_list,
        device='cpu'
    )
    
    print("\n" + "=" * 80)
    print("하이퍼파라미터 튜닝 완료!")
    print("=" * 80)
    print(f"\n최고 성능 모델이 {model_save_path}/best_model 에 저장되었습니다.")
    print("이제 FastAPI 서버에서 이 모델을 사용할 수 있습니다.")


if __name__ == "__main__":
    main()
