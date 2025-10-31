#!/usr/bin/env python3
"""
AWS EC2 최적화 모델 학습 스크립트

특징:
- 메모리 사용량 모니터링
- 배치 크기 자동 조정
- GPU/CPU 자동 감지
- 학습 진행률 상세 출력
- 모델 저장 및 백업
"""

import os
import sys
import time
import psutil
import torch
import pandas as pd
from hybrid_recommender import HybridRecommendationSystem

def get_system_info():
    """시스템 정보 출력"""
    print("=== 시스템 정보 ===")
    print(f"CPU 코어 수: {psutil.cpu_count()}")
    print(f"메모리: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"사용 가능 메모리: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        device = "cuda"
    else:
        print("GPU: 없음 (CPU 사용)")
        device = "cpu"
    
    return device

def optimize_batch_size(available_memory_gb, device):
    """메모리에 따른 배치 크기 자동 조정"""
    if device == "cuda":
        # GPU 메모리 기반
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory_gb >= 16:
            return 1024
        elif gpu_memory_gb >= 8:
            return 512
        else:
            return 256
    else:
        # CPU 메모리 기반
        if available_memory_gb >= 16:
            return 1024
        elif available_memory_gb >= 8:
            return 512
        elif available_memory_gb >= 4:
            return 256
        else:
            return 128

def monitor_training_progress(start_time, epoch, total_epochs):
    """학습 진행률 모니터링"""
    elapsed = time.time() - start_time
    progress = (epoch + 1) / total_epochs
    eta = elapsed / progress - elapsed if progress > 0 else 0
    
    print(f"진행률: {progress*100:.1f}% | "
          f"경과 시간: {elapsed/60:.1f}분 | "
          f"예상 남은 시간: {eta/60:.1f}분 | "
          f"메모리 사용량: {psutil.virtual_memory().percent:.1f}%")

def main():
    print("=== AWS EC2 최적화 모델 학습 ===")
    
    # 시스템 정보 확인
    device = get_system_info()
    available_memory = psutil.virtual_memory().available / (1024**3)
    
    # 배치 크기 최적화
    optimal_batch_size = optimize_batch_size(available_memory, device)
    print(f"\n최적화된 배치 크기: {optimal_batch_size}")
    
    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data", "splitty_recommendation_data_1")
    model_path = os.path.join(current_dir, "saved_models")
    backup_path = os.path.join(current_dir, "model_backup")
    
    # 백업 디렉토리 생성
    os.makedirs(backup_path, exist_ok=True)
    
    print(f"\n데이터 경로: {data_path}")
    print(f"모델 저장 경로: {model_path}")
    print(f"백업 경로: {backup_path}")
    
    # 데이터 존재 확인
    train_file = os.path.join(data_path, "user_item_train.csv")
    if not os.path.exists(train_file):
        print(f"\n❌ 오류: 학습 데이터를 찾을 수 없습니다: {train_file}")
        return
    
    # 데이터 크기 확인
    train_df = pd.read_csv(train_file)
    data_size_mb = train_df.memory_usage(deep=True).sum() / (1024**2)
    print(f"\n데이터 크기: {len(train_df):,}행, {data_size_mb:.1f}MB")
    
    # 학습 파라미터 조정
    n_users = train_df['user_id'].nunique()
    n_items = train_df['item_id'].nunique()
    
    # 데이터 크기에 따른 에포크 조정
    if len(train_df) > 100000:
        mf_epochs = 80
        tt_epochs = 30
    elif len(train_df) > 50000:
        mf_epochs = 100
        tt_epochs = 40
    else:
        mf_epochs = 120
        tt_epochs = 50
    
    print(f"\n학습 설정:")
    print(f"- 사용자 수: {n_users:,}")
    print(f"- 아이템 수: {n_items:,}")
    print(f"- MF 에포크: {mf_epochs}")
    print(f"- Two-Tower 에포크: {tt_epochs}")
    print(f"- 배치 크기: {optimal_batch_size}")
    print(f"- 디바이스: {device.upper()}")
    
    # 하이브리드 추천 시스템 초기화
    print(f"\n🚀 하이브리드 추천 시스템 초기화 중...")
    recommender = HybridRecommendationSystem(device=device)
    
    try:
        # 학습 시작
        start_time = time.time()
        
        print("\n📊 데이터 로드 중...")
        recommender.load_data(data_path)
        
        print("\n🤖 모델 학습 시작...")
        
        # Matrix Factorization 파라미터 조정
        class CustomHybridRecommender(HybridRecommendationSystem):
            def train_models(self, mf_factors=50, epochs=50, batch_size=1024):
                """EC2 최적화된 학습 메서드"""
                print("=== EC2 최적화 하이브리드 모델 학습 ===")
                
                # 1단계: Implicit Matrix Factorization
                print(f"\n1단계: Implicit Matrix Factorization 학습")
                from models import ImplicitMatrixFactorization
                
                self.matrix_factorization = ImplicitMatrixFactorization(
                    n_factors=mf_factors,
                    learning_rate=0.01,
                    regularization=0.01,
                    n_epochs=mf_epochs,  # EC2 최적화
                    negative_samples=3   # 메모리 절약을 위해 줄임
                )
                
                mf_start = time.time()
                self.matrix_factorization.fit(self.train_data, use_negative_sampling=True)
                mf_time = time.time() - mf_start
                print(f"MF 학습 완료 ({mf_time/60:.1f}분)")
                
                # 2단계: Two-Tower 모델
                print(f"\n2단계: Two-Tower 모델 학습")
                
                n_users = self.train_data['user_id'].nunique()
                n_items = self.train_data['item_id'].nunique()
                n_categories = self.train_data['category'].nunique()
                
                from models import TwoTowerModel, TwoTowerTrainer
                
                # 모델 크기 조정 (메모리 절약)
                embedding_dim = min(64, max(32, n_users // 100))
                hidden_dims = [min(128, embedding_dim * 2), embedding_dim]
                
                two_tower_model = TwoTowerModel(
                    n_users=n_users,
                    n_items=n_items,
                    n_categories=n_categories,
                    embedding_dim=embedding_dim,
                    hidden_dims=hidden_dims
                )
                
                self.two_tower_trainer = TwoTowerTrainer(two_tower_model, device=device)
                
                tt_start = time.time()
                self.two_tower_trainer.fit(
                    train_df=self.train_data,
                    epochs=tt_epochs,  # EC2 최적화
                    batch_size=optimal_batch_size,
                    lr=0.001
                )
                tt_time = time.time() - tt_start
                print(f"Two-Tower 학습 완료 ({tt_time/60:.1f}분)")
                
                self.is_trained = True
                total_time = time.time() - start_time
                print(f"\n✅ 전체 학습 완료 ({total_time/60:.1f}분)")
        
        # 커스텀 학습 실행
        custom_recommender = CustomHybridRecommender(device=device)
        custom_recommender.train_data = recommender.train_data
        custom_recommender.item_meta = recommender.item_meta
        
        custom_recommender.train_models(
            mf_factors=50,
            epochs=tt_epochs,
            batch_size=optimal_batch_size
        )
        
        # 모델 저장
        print("\n💾 모델 저장 중...")
        custom_recommender.save_models(model_path)
        
        # 백업 저장
        import shutil
        if os.path.exists(model_path):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(backup_path, f"model_{timestamp}")
            shutil.copytree(model_path, backup_dir)
            print(f"백업 저장: {backup_dir}")
        
        # 시스템 통계
        print("\n📈 시스템 통계:")
        stats = custom_recommender.get_system_stats()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        
        # 간단한 테스트
        print("\n🧪 간단한 추천 테스트:")
        sample_users = custom_recommender.train_data['user_id'].unique()[:3]
        
        for user_id in sample_users:
            try:
                recommendations = custom_recommender.get_recommendations(
                    user_id=user_id, top_k=50, top_n=5
                )
                print(f"  사용자 {user_id}: {len(recommendations)}개 추천")
            except Exception as e:
                print(f"  사용자 {user_id}: 추천 실패 - {str(e)}")
        
        total_time = time.time() - start_time
        print(f"\n🎉 EC2 학습 완료! (총 소요시간: {total_time/60:.1f}분)")
        
    except KeyboardInterrupt:
        print("\n⚠️  사용자 중단")
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\n최종 메모리 사용량: {psutil.virtual_memory().percent:.1f}%")

if __name__ == "__main__":
    main()