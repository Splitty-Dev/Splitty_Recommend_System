#!/usr/bin/env python3
"""
하이브리드 추천 시스템 학습 및 테스트 스크립트

이 스크립트는:
1. 학습 데이터를 로드
2. Matrix Factorization과 Two-Tower 모델을 학습
3. 모델을 저장
4. 테스트 추천을 실행

사용법:
python train_and_test.py
"""

import os
import sys
import pandas as pd
from hybrid_recommender import HybridRecommendationSystem

def main():
    print("=== 하이브리드 추천 시스템 학습 및 테스트 ===")
    
    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data", "splitty_recommendation_data_1")
    model_path = os.path.join(current_dir, "saved_models")
    
    print(f"데이터 경로: {data_path}")
    print(f"모델 저장 경로: {model_path}")
    
    # 데이터 존재 확인
    train_file = os.path.join(data_path, "user_item_train.csv")
    if not os.path.exists(train_file):
        print(f"오류: 학습 데이터를 찾을 수 없습니다: {train_file}")
        print("먼저 데이터 생성 스크립트를 실행하여 train 데이터를 생성하세요.")
        return
    
    # 하이브리드 추천 시스템 초기화
    print("\n1. 하이브리드 추천 시스템 초기화...")
    recommender = HybridRecommendationSystem(device='cpu')
    
    # 데이터 로드
    print("\n2. 데이터 로드...")
    recommender.load_data(data_path)
    
    # 모델 학습 (네거티브 샘플링 포함)
    print("\n3. 모델 학습 (네거티브 샘플링 포함)...")
    recommender.train_models(
        mf_factors=50,
        epochs=30,  # Two-Tower epochs 
        batch_size=512
    )
    
    # 모델 저장
    print("\n4. 모델 저장...")
    recommender.save_models(model_path)
    
    # 시스템 통계 출력
    print("\n5. 시스템 통계:")
    stats = recommender.get_system_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 테스트 추천
    print("\n6. 테스트 추천 실행...")
    
    # 학습 데이터에서 몇 개 사용자 샘플링
    sample_users = recommender.train_data['user_id'].unique()[:5]
    
    for user_id in sample_users:
        print(f"\n--- 사용자 {user_id} 추천 결과 ---")
        
        try:
            # 하이브리드 추천
            recommendations = recommender.get_recommendations(
                user_id=user_id,
                top_k=250,
                top_n=10
            )
            
            print(f"추천 개수: {len(recommendations)}")
            for i, rec in enumerate(recommendations[:5]):  # 상위 5개만 출력
                print(f"  {i+1}. {rec['title']} (점수: {rec['final_score']:.3f})")
                
        except Exception as e:
            print(f"  추천 실패: {str(e)}")
    
    # 인기 아이템 테스트
    print(f"\n--- 인기 아이템 추천 ---")
    try:
        popular_items = recommender.get_popular_recommendations(top_n=10)
        print(f"인기 아이템 개수: {len(popular_items)}")
        for i, item in enumerate(popular_items[:5]):
            print(f"  {i+1}. {item['title']} (점수: {item['final_score']:.3f})")
    except Exception as e:
        print(f"  인기 아이템 추천 실패: {str(e)}")
    
    print("\n=== 학습 및 테스트 완료 ===")
    print(f"모델이 {model_path}에 저장되었습니다.")
    print("이제 FastAPI 서버를 시작할 수 있습니다: python main.py")

if __name__ == "__main__":
    main()