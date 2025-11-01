#!/bin/bash

# Splitty 추천 시스템 API 테스트 스크립트
# Usage: bash test_api.sh

API_URL="http://3.25.74.202:8000/api/recommend"

echo "=========================================="
echo "Splitty Recommendation API Test Cases"
echo "=========================================="
echo ""

# 기존 사용자 테스트 (user_id: "1")
echo "=== 기존 사용자 (user_id: 1) 테스트 ==="
echo ""

echo "[Test 1] 유저1이 10개 아이템 바라는 경우"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "1", "top_n": 10}'
echo -e "\n"

echo "[Test 2] 유저1이 50개 아이템 바라는 경우"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "1", "top_n": 50}'
echo -e "\n"

echo "[Test 3] 유저1이 카테고리 1의 아이템을 10개 바라는 경우"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "1", "top_n": 10, "categoryId": 1}'
echo -e "\n"

echo "[Test 4] 유저1이 카테고리 2의 아이템을 10개 바라는 경우"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "1", "top_n": 10, "categoryId": 2}'
echo -e "\n"

echo "[Test 5] 유저1이 카테고리 3의 아이템을 10개 바라는 경우"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "1", "top_n": 10, "categoryId": 3}'
echo -e "\n"

echo "[Test 6] 유저1이 카테고리 4의 아이템을 10개 바라는 경우"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "1", "top_n": 10, "categoryId": 4}'
echo -e "\n"

echo "[Test 7] 유저1이 카테고리 5의 아이템을 10개 바라는 경우"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "1", "top_n": 10, "categoryId": 5}'
echo -e "\n"

echo "[Test 8] 유저1이 카테고리 6의 아이템을 10개 바라는 경우"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "1", "top_n": 10, "categoryId": 6}'
echo -e "\n"

echo "[Test 9] 유저1이 목록 중에서 상위 10개 아이템을 원하는 경우 (목록에 아이템이 5개)"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "1", "top_n": 10, "available_items": [15, 42, 89, 156, 203]}'
echo -e "\n"

echo "[Test 10] 유저1이 목록 중에서 상위 10개 아이템을 원하는 경우 (목록에 아이템이 10개)"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "1", "top_n": 10, "available_items": [15, 16, 17, 42, 78, 88, 89, 156, 203, 233]}'
echo -e "\n"

echo "[Test 11] 유저1이 목록 중에서 상위 10개 아이템을 원하는 경우 (목록에 아이템이 20개)"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "1", "top_n": 10, "available_items": [15, 16, 17, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 42, 78, 88, 89, 156, 203, 233]}'
echo -e "\n"

echo "[Test 12] 유저1이 목록 중에서 카테고리 1인 상위 10개 아이템을 원하는 경우 (목록에 아이템이 20개)"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "1", "top_n": 10, "categoryId": 1, "available_items": [15, 16, 17, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 42, 78, 88, 89, 156, 203, 233]}'
echo -e "\n"

echo ""
echo "=== 신규 사용자 (user_id: newUser) 테스트 ==="
echo ""

echo "[Test 13] 신규 유저가 10개 아이템 바라는 경우"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "newUser", "top_n": 10}'
echo -e "\n"

echo "[Test 14] 신규 유저가 50개 아이템 바라는 경우"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "newUser", "top_n": 50}'
echo -e "\n"

echo "[Test 15] 신규 유저가 카테고리 1의 아이템을 10개 바라는 경우"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "newUser", "top_n": 10, "categoryId": 1}'
echo -e "\n"

echo "[Test 16] 신규 유저가 카테고리 2의 아이템을 10개 바라는 경우"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "newUser", "top_n": 10, "categoryId": 2}'
echo -e "\n"

echo "[Test 17] 신규 유저가 카테고리 3의 아이템을 10개 바라는 경우"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "newUser", "top_n": 10, "categoryId": 3}'
echo -e "\n"

echo "[Test 18] 신규 유저가 카테고리 4의 아이템을 10개 바라는 경우"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "newUser", "top_n": 10, "categoryId": 4}'
echo -e "\n"

echo "[Test 19] 신규 유저가 카테고리 5의 아이템을 10개 바라는 경우"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "newUser", "top_n": 10, "categoryId": 5}'
echo -e "\n"

echo "[Test 20] 신규 유저가 카테고리 6의 아이템을 10개 바라는 경우"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "newUser", "top_n": 10, "categoryId": 6}'
echo -e "\n"

echo "[Test 21] 신규 유저가 목록 중에서 상위 10개 아이템을 원하는 경우 (목록에 아이템이 5개)"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "newUser", "top_n": 10, "available_items": [15, 42, 89, 156, 203]}'
echo -e "\n"

echo "[Test 22] 신규 유저가 목록 중에서 상위 10개 아이템을 원하는 경우 (목록에 아이템이 10개)"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "newUser", "top_n": 10, "available_items": [15, 16, 17, 42, 78, 88, 89, 156, 203, 233]}'
echo -e "\n"

echo "[Test 23] 신규 유저가 목록 중에서 상위 10개 아이템을 원하는 경우 (목록에 아이템이 20개)"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "newUser", "top_n": 10, "available_items": [15, 16, 17, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 42, 78, 88, 89, 156, 203, 233]}'
echo -e "\n"

echo "[Test 24] 신규 유저가 목록 중에서 카테고리 1인 상위 10개 아이템을 원하는 경우 (목록에 아이템이 20개)"
curl -X POST $API_URL -H "Content-Type: application/json" -d '{"user_id": "newUser", "top_n": 10, "categoryId": 1, "available_items": [15, 16, 17, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 42, 78, 88, 89, 156, 203, 233]}'
echo -e "\n"

echo "=========================================="
echo "모든 테스트 완료!"
echo "=========================================="
