# API 테스트 가이드

## 테스트 스크립트 사용법

### 전체 테스트 실행
```bash
cd server_fastAPI
bash test_api.sh
```

### 특정 테스트만 실행하고 싶은 경우
파일을 열어서 원하는 테스트 케이스만 복사해서 실행하세요.

## 테스트 케이스 목록

### 기존 사용자 테스트 (user_id: "1")
- **Test 1**: 10개 아이템 요청
- **Test 2**: 50개 아이템 요청
- **Test 3-8**: 카테고리별 10개 아이템 요청 (카테고리 1-6)
- **Test 9**: available_items 5개 중에서 10개 요청
- **Test 10**: available_items 10개 중에서 10개 요청
- **Test 11**: available_items 20개 중에서 10개 요청
- **Test 12**: available_items 20개 + 카테고리 1 필터링

### 신규 사용자 테스트 (user_id: "newUser")
- **Test 13**: 10개 아이템 요청
- **Test 14**: 50개 아이템 요청
- **Test 15-20**: 카테고리별 10개 아이템 요청 (카테고리 1-6)
- **Test 21**: available_items 5개 중에서 10개 요청
- **Test 22**: available_items 10개 중에서 10개 요청
- **Test 23**: available_items 20개 중에서 10개 요청
- **Test 24**: available_items 20개 + 카테고리 1 필터링

## 카테고리 매핑
- **1**: 식품
- **2**: 생활/주방
- **3**: 뷰티/미용
- **4**: 패션
- **5**: 건강/운동
- **6**: 유아

## API 엔드포인트
```
POST http://3.25.74.202:8000/api/recommend
```

## 요청 형식
```json
{
  "user_id": "string",
  "top_n": 10,
  "categoryId": 1,  // 선택적 (1-6)
  "available_items": [15, 42, 89]  // 선택적
}
```

## 응답 형식
```json
{
  "user_id": "string",
  "items": [
    {"item_id": 297, "rank": 1},
    {"item_id": 418, "rank": 2}
  ]
}
```

## 주의사항
- 기존 사용자는 개인화 추천을 받습니다
- 신규 사용자는 인기 아이템 추천을 받습니다
- available_items가 제공되면 해당 아이템 중에서만 추천합니다
- categoryId가 제공되면 해당 카테고리의 아이템만 추천합니다
- 두 필터를 동시에 사용할 수 있습니다
