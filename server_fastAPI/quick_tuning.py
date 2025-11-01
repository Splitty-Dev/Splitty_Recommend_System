#!/usr/bin/env python3
"""
빠른 하이퍼파라미터 튜닝 스크립트 (테스트용)

작은 Grid로 빠르게 테스트합니다.
실제 사용 시 hyperparameter_tuning.py를 사용하세요.

사용법:
python quick_tuning.py
"""

import os
import sys
from hyperparameter_tuning import grid_search_hyperparameters

def main():
    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data", "splitty_recommendation_data_1")
    model_save_path = os.path.join(current_dir, "saved_models")
    
    os.makedirs(model_save_path, exist_ok=True)
    
    print("=" * 80)
    print("빠른 하이퍼파라미터 튜닝 (테스트용)")
    print("=" * 80)
    print(f"데이터 경로: {data_path}")
    print(f"모델 저장 경로: {model_save_path}")
    print("\n⚠️  주의: 이것은 테스트용 작은 Grid입니다.")
    print("실제 사용 시 hyperparameter_tuning.py를 사용하세요.\n")
    
    # 작은 Grid Search 파라미터 정의 (빠른 테스트용)
    param_grid = {
        'mf_factors': [30, 50],          # 2가지
        'epochs': [20, 30],               # 2가지
        'batch_size': [512],              # 1가지 (고정)
        'top_k': [200, 250],              # 2가지
        'top_n': [50],                    # 1가지 (고정)
    }
    
    # 총 조합: 2 * 2 * 1 * 2 * 1 = 8가지
    
    # 평가할 K 값들
    eval_k_list = [5, 10]
    
    print(f"총 {2 * 2 * 1 * 2 * 1}가지 조합을 테스트합니다.")
    print("예상 소요 시간: 약 20-30분\n")
    
    # Grid Search 실행
    best_params, best_metrics, all_results = grid_search_hyperparameters(
        data_path=data_path,
        model_save_path=model_save_path,
        param_grid=param_grid,
        eval_k_list=eval_k_list,
        device='cpu'
    )
    
    print("\n" + "=" * 80)
    print("빠른 튜닝 완료!")
    print("=" * 80)
    print("\n더 정밀한 튜닝을 원하시면 hyperparameter_tuning.py를 실행하세요.")


if __name__ == "__main__":
    main()
