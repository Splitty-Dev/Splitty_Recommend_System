"""
S3 로그를 가져와서 모델을 재학습하는 전체 파이프라인
"""
import os
import sys
from datetime import datetime

# 모듈 import
from s3_log_fetcher import S3LogFetcher
from log_data_transformer import LogDataTransformer
from data_merger import DataMerger
from hybrid_recommender import HybridRecommender


class IncrementalTrainingPipeline:
    """증분 학습 파이프라인"""
    
    def __init__(
        self,
        bucket_name: str = "splitty-recommendation-log-bucket",
        data_dir: str = "../data/splitty_recommendation_data_1",
        model_dir: str = "./saved_models"
    ):
        """
        Args:
            bucket_name: S3 버킷 이름
            data_dir: 학습 데이터 디렉토리
            model_dir: 모델 저장 디렉토리
        """
        self.bucket_name = bucket_name
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        # 모듈 초기화
        self.log_fetcher = S3LogFetcher(bucket_name)
        self.transformer = LogDataTransformer()
        self.merger = DataMerger(data_dir)
        self.recommender = None
    
    def run(
        self,
        max_log_files: int = 10,
        retrain: bool = True,
        backup: bool = True,
        s3_prefix: str = ""
    ):
        """
        전체 파이프라인을 실행합니다.
        
        Args:
            max_log_files: 가져올 최대 로그 파일 개수
            retrain: 모델 재학습 여부
            backup: 기존 데이터 백업 여부
            s3_prefix: S3 파일 경로 prefix (예: "logs/2025/11/")
        """
        print("="*80)
        print("증분 학습 파이프라인 시작")
        print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if s3_prefix:
            print(f"S3 경로: {self.bucket_name}/{s3_prefix}")
        print("="*80)
        
        try:
            # 1. S3에서 로그 가져오기
            print("\n[1/6] S3에서 로그 가져오기...")
            
            # prefix 설정 (자동으로 하위 디렉토리까지 검색)
            self.log_fetcher.list_log_files(
                prefix=s3_prefix,
                max_files=max_log_files * 3  # 필터링 고려
            )
            
            logs = self.log_fetcher.fetch_latest_logs(
                max_files=max_log_files,
                skip_processed=True
            )
            
            if not logs:
                print("가져올 로그가 없습니다. 파이프라인을 종료합니다.")
                return
            
            print(f"✓ {len(logs)}개의 로그 레코드를 가져왔습니다.")
            
            # 2. 로그를 학습 데이터로 변환
            print("\n[2/6] 로그를 학습 데이터로 변환...")
            new_training_data = self.transformer.transform_logs_to_training_data(logs)
            
            if len(new_training_data) == 0:
                print("변환된 학습 데이터가 없습니다. 파이프라인을 종료합니다.")
                return
            
            print(f"✓ {len(new_training_data)}개의 학습 레코드로 변환되었습니다.")
            
            # 3. 기존 데이터 로드
            print("\n[3/6] 기존 학습 데이터 로드...")
            train_df, test_df, val_df = self.merger.load_existing_data()
            print(f"✓ 기존 데이터 로드 완료")
            print(f"  Train: {len(train_df)}개, Val: {len(val_df)}개, Test: {len(test_df)}개")
            
            # 4. 데이터 병합 (train에만 추가, test/val 유지)
            print("\n[4/6] 새 데이터를 train에 추가 (test/val은 유지)...")
            merged_train = self.merger.merge_new_data(
                existing_train=train_df,
                new_data=new_training_data,
                deduplicate=True,
                merge_strategy="train_only"
            )
            
            # 병합 전후 비교
            self.merger.print_comparison(train_df, merged_train)
            
            # 5. 데이터 저장 (test/val은 그대로)
            print("\n[5/6] 새 train 데이터 저장 (test/val 유지)...")
            self.merger.save_merged_data(merged_train, val_df, test_df, backup=backup)
            print(f"✓ 데이터 저장 완료")
            print(f"  최종 - Train: {len(merged_train)}개, Val: {len(val_df)}개, Test: {len(test_df)}개")
            
            # 6. 모델 재학습
            if retrain:
                print("\n[6/6] 모델 재학습...")
                self._retrain_model()
                print(f"✓ 모델 재학습 완료")
            else:
                print("\n[6/6] 모델 재학습 스킵 (retrain=False)")
            
            print("\n" + "="*80)
            print("증분 학습 파이프라인 완료!")
            print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            
        except Exception as e:
            print(f"\n❌ 파이프라인 실행 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _retrain_model(self):
        """
        새로운 데이터로 모델을 재학습합니다.
        """
        print("  하이브리드 추천 모델 초기화...")
        self.recommender = HybridRecommender(device='cpu')
        
        print("  데이터 로드 중...")
        self.recommender.load_data(self.data_dir)
        
        print("  모델 학습 중... (이 과정은 시간이 걸릴 수 있습니다)")
        self.recommender.train_models(
            mf_factors=50,
            epochs=30,
            batch_size=512
        )
        
        # 기존 모델 백업
        if os.path.exists(self.model_dir):
            from datetime import datetime
            backup_suffix = datetime.now().strftime("_%Y%m%d_%H%M%S.backup")
            backup_dir = self.model_dir + backup_suffix
            os.rename(self.model_dir, backup_dir)
            print(f"  기존 모델 백업: {backup_dir}")
        
        # 새 모델 저장
        os.makedirs(self.model_dir, exist_ok=True)
        self.recommender.save_models(self.model_dir)
        print(f"  새 모델 저장: {self.model_dir}")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='S3 로그 기반 증분 학습 파이프라인')
    parser.add_argument(
        '--bucket',
        type=str,
        default='splitty-recommendation-log-bucket',
        help='S3 버킷 이름'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data/splitty_recommendation_data_1',
        help='학습 데이터 디렉토리'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='./saved_models',
        help='모델 저장 디렉토리'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=10,
        help='가져올 최대 로그 파일 개수'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='',
        help='S3 파일 경로 prefix (예: "logs/2025/11/")'
    )
    parser.add_argument(
        '--no-retrain',
        action='store_true',
        help='모델 재학습을 스킵합니다 (데이터 병합만 수행)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='기존 데이터 백업을 스킵합니다'
    )
    
    args = parser.parse_args()
    
    # 파이프라인 실행
    pipeline = IncrementalTrainingPipeline(
        bucket_name=args.bucket,
        data_dir=args.data_dir,
        model_dir=args.model_dir
    )
    
    pipeline.run(
        max_log_files=args.max_files,
        retrain=not args.no_retrain,
        backup=not args.no_backup,
        s3_prefix=args.prefix
    )


if __name__ == "__main__":
    main()
