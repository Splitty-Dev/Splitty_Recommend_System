"""
S3에서 추천 로그를 가져오는 모듈
"""
import boto3
import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import os


class S3LogFetcher:
    """S3 버킷에서 추천 로그를 가져오는 클래스"""
    
    def __init__(
        self, 
        bucket_name: str = "splitty-recommendation-log-bucket",
        processed_files_path: str = "./processed_log_files.json"
    ):
        """
        Args:
            bucket_name: S3 버킷 이름
            processed_files_path: 처리된 파일 목록을 저장할 경로
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        self.processed_files_path = processed_files_path
        self.processed_files = self._load_processed_files()
        
    def _load_processed_files(self) -> set:
        """
        처리된 파일 목록을 로드합니다.
        
        Returns:
            처리된 파일 키의 set
        """
        if os.path.exists(self.processed_files_path):
            try:
                with open(self.processed_files_path, 'r') as f:
                    data = json.load(f)
                    return set(data.get('processed_files', []))
            except Exception as e:
                print(f"처리된 파일 목록 로드 실패: {str(e)}")
                return set()
        return set()
    
    def _save_processed_files(self):
        """
        처리된 파일 목록을 저장합니다.
        """
        try:
            data = {
                'processed_files': list(self.processed_files),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.processed_files_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"처리된 파일 목록 저장: {len(self.processed_files)}개")
        except Exception as e:
            print(f"처리된 파일 목록 저장 실패: {str(e)}")
    
    def mark_as_processed(self, file_key: str):
        """
        파일을 처리됨으로 표시합니다.
        
        Args:
            file_key: S3 파일 키
        """
        self.processed_files.add(file_key)
        self._save_processed_files()
    
    def is_processed(self, file_key: str) -> bool:
        """
        파일이 이미 처리되었는지 확인합니다.
        
        Args:
            file_key: S3 파일 키
            
        Returns:
            처리 여부
        """
        return file_key in self.processed_files
        
    def list_log_files(
        self, 
        prefix: str = "", 
        max_files: int = 100,
        file_extension: str = ".json",
        recursive: bool = True
    ) -> List[str]:
        """
        S3 버킷에서 로그 파일 목록을 가져옵니다.
        
        Args:
            prefix: 파일 경로 prefix (예: "logs/2025/11/")
            max_files: 최대 파일 개수
            file_extension: 파일 확장자 필터 (예: ".json")
            recursive: 하위 디렉토리까지 재귀적으로 검색
            
        Returns:
            파일 키 리스트
        """
        try:
            all_files = []
            continuation_token = None
            
            while len(all_files) < max_files:
                # S3 API 호출
                params = {
                    'Bucket': self.bucket_name,
                    'Prefix': prefix,
                    'MaxKeys': min(1000, max_files * 2)  # 한 번에 많이 가져오기
                }
                
                if continuation_token:
                    params['ContinuationToken'] = continuation_token
                
                response = self.s3_client.list_objects_v2(**params)
                
                if 'Contents' not in response:
                    break
                
                # 파일 필터링 (확장자 체크, 디렉토리 제외)
                for obj in response['Contents']:
                    key = obj['Key']
                    
                    # 디렉토리 스킵 (끝이 /인 경우)
                    if key.endswith('/'):
                        continue
                    
                    # 확장자 필터링
                    if file_extension and not key.endswith(file_extension):
                        continue
                    
                    all_files.append(key)
                    
                    if len(all_files) >= max_files:
                        break
                
                # 다음 페이지가 있으면 계속
                if response.get('IsTruncated'):
                    continuation_token = response.get('NextContinuationToken')
                else:
                    break
            
            # 최신 파일부터 정렬 (LastModified 기준)
            all_files = all_files[:max_files]
            
            print(f"발견된 로그 파일: {len(all_files)}개 (prefix: '{prefix if prefix else '(루트)'}')")
            if all_files and len(all_files) > 0:
                print(f"  예시: {all_files[0]}")
            
            return all_files
            
        except Exception as e:
            print(f"S3 파일 목록 조회 실패: {str(e)}")
            return []
    
    def fetch_log_file(self, file_key: str) -> Optional[List[Dict]]:
        """
        S3에서 단일 로그 파일을 다운로드하고 파싱합니다.
        
        Args:
            file_key: S3 파일 키
            
        Returns:
            로그 레코드 리스트 또는 None
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=file_key
            )
            
            content = response['Body'].read().decode('utf-8')
            logs = json.loads(content)
            
            print(f"로그 파일 로드 성공: {file_key} ({len(logs)}개 레코드)")
            return logs
            
        except Exception as e:
            print(f"로그 파일 로드 실패 ({file_key}): {str(e)}")
            return None
    
    def fetch_logs_by_date(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        prefix: str = "",
        skip_processed: bool = True
    ) -> List[Dict]:
        """
        날짜 범위로 로그를 가져옵니다 (이미 처리된 파일은 스킵).
        
        Args:
            start_date: 시작 날짜 (None이면 제한 없음)
            end_date: 종료 날짜 (None이면 오늘)
            prefix: 파일 경로 prefix
            skip_processed: 처리된 파일 스킵 여부 (기본값: True)
            
        Returns:
            모든 로그 레코드의 리스트
        """
        all_logs = []
        processed_count = 0
        fetched_count = 0
        
        # 파일 목록 가져오기
        file_keys = self.list_log_files(prefix=prefix)
        
        for file_key in file_keys:
            # 이미 처리된 파일 스킵
            if skip_processed and self.is_processed(file_key):
                processed_count += 1
                continue
            
            logs = self.fetch_log_file(file_key)
            if logs:
                # 날짜 필터링
                filtered_logs = self._filter_by_timestamp(logs, start_date, end_date)
                if filtered_logs:
                    all_logs.extend(filtered_logs)
                    # 파일을 처리됨으로 표시
                    self.mark_as_processed(file_key)
                    fetched_count += 1
        
        print(f"총 {len(all_logs)}개의 로그 레코드를 가져왔습니다.")
        print(f"  처리된 파일 스킵: {processed_count}개")
        print(f"  새로 가져온 파일: {fetched_count}개")
        return all_logs
    
    def fetch_latest_logs(
        self, 
        max_files: int = 10,
        skip_processed: bool = True
    ) -> List[Dict]:
        """
        최신 로그 파일들을 가져옵니다 (이미 처리된 파일은 스킵).
        
        Args:
            max_files: 최대 파일 개수
            skip_processed: 처리된 파일 스킵 여부 (기본값: True)
            
        Returns:
            로그 레코드 리스트
        """
        all_logs = []
        processed_count = 0
        fetched_count = 0
        
        # 모든 파일 목록 가져오기 (더 많이 가져와서 필터링)
        file_keys = self.list_log_files(max_files=max_files * 3)
        
        # 최신 파일부터 처리
        for file_key in sorted(file_keys, reverse=True):
            # 이미 처리된 파일 스킵
            if skip_processed and self.is_processed(file_key):
                processed_count += 1
                continue
            
            # 필요한 개수만큼만 가져오기
            if fetched_count >= max_files:
                break
            
            logs = self.fetch_log_file(file_key)
            if logs:
                all_logs.extend(logs)
                # 파일을 처리됨으로 표시
                self.mark_as_processed(file_key)
                fetched_count += 1
        
        print(f"새로운 로그 {len(all_logs)}개를 가져왔습니다.")
        print(f"  처리된 파일 스킵: {processed_count}개")
        print(f"  새로 가져온 파일: {fetched_count}개")
        return all_logs
    
    def _filter_by_timestamp(
        self,
        logs: List[Dict],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[Dict]:
        """
        타임스탬프로 로그를 필터링합니다.
        
        Args:
            logs: 로그 레코드 리스트
            start_date: 시작 날짜
            end_date: 종료 날짜
            
        Returns:
            필터링된 로그 리스트
        """
        if not start_date and not end_date:
            return logs
        
        filtered = []
        for log in logs:
            if 'timestamp' not in log:
                continue
            
            # 밀리초 타임스탬프를 datetime으로 변환
            log_time = datetime.fromtimestamp(log['timestamp'] / 1000)
            
            if start_date and log_time < start_date:
                continue
            if end_date and log_time > end_date:
                continue
            
            filtered.append(log)
        
        return filtered
    
    def save_logs_locally(self, logs: List[Dict], output_path: str):
        """
        로그를 로컬 파일로 저장합니다.
        
        Args:
            logs: 로그 레코드 리스트
            output_path: 저장할 파일 경로
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
            
            print(f"로그를 로컬에 저장했습니다: {output_path}")
            
        except Exception as e:
            print(f"로그 저장 실패: {str(e)}")


# 사용 예시
if __name__ == "__main__":
    # S3 로그 fetcher 초기화
    fetcher = S3LogFetcher()
    
    # 최신 로그 10개 파일 가져오기
    logs = fetcher.fetch_latest_logs(max_files=10)
    
    print(f"\n샘플 로그 (처음 3개):")
    for i, log in enumerate(logs[:3]):
        print(f"{i+1}. {log}")
    
    # 로컬에 저장
    if logs:
        fetcher.save_logs_locally(logs, "./data/raw_logs.json")
