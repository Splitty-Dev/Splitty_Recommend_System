#!/usr/bin/env python3
"""
Splitty 추천 시스템 학습 데이터 생성 스크립트

생성되는 파일들:
- user_item_train.csv: 학습 데이터셋
- user_item_val.csv: 검증 데이터셋
- user_item_test.csv: 테스트 데이터셋
- encoders.json: 범주형 변수 인코더 정보
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import defaultdict


# 백엔드 카테고리 매핑 (int → 한글명)
CATEGORY_MAPPING = {
    1: "식품",
    2: "생활/주방",
    3: "뷰티/미용",
    4: "패션",
    5: "건강/운동",
    6: "유아"
}


def make_synthetic_data(n_users=200, n_items=500, n_events=5000, seed=42):
    """
    추천 시스템 학습을 위한 가상 데이터를 생성하는 함수
    
    Args:
        n_users: 생성할 사용자 수
        n_items: 생성할 아이템 수
        n_events: 생성할 이벤트 수
        seed: 랜덤 시드
    
    Returns:
        df: 이벤트 데이터프레임
        item_meta: 아이템 메타데이터
        user_ids: 사용자 ID 리스트
        item_ids: 아이템 ID 리스트
    """
    np.random.seed(seed)
    user_ids = [f"user_{i}" for i in range(1, n_users + 1)]
    item_ids = [f"item_{i}" for i in range(1, n_items + 1)]

    # === 카테고리별 브랜드 및 제품 목록 ===
    category_data = {
        1: {  # 식품
            "brands": ["삼다수", "아이시스", "펩시", "코카콜라", "칠성"],
            "products": ["생수", "콜라", "사이다", "제로콜라", "생수 번들", "묶음 캔", "캔", "병", "큰병"]
        },
        2: {  # 생활/주방
            "brands": ["비트", "테크", "퍼실", "다우니"],
            "products": ["섬유유연제", "세제", "캡슐세제", "시트형", "시트형 세제", "캡슐형"]
        },
        3: {  # 뷰티/미용
            "brands": ["피지오겔", "닥터지", "오휘", "네이처리퍼블릭", "과일나라"],
            "products": ["수분크림", "앰플", "로션", "자외선차단제", "선크림", "썬크림", "알로에"]
        },
        4: {  # 패션
            "brands": ["자라", "유니클로", "H&M", "무신사", "에이블리"],
            "products": ["티셔츠", "바지", "원피스", "자켓", "후드", "맨투맨", "셔츠", "니트", "코트"]
        },
        5: {  # 건강/운동
            "brands": ["맛있닭", "랭커", "잇메이트", "랭킹닭컴", "셀렉스", "칼로바이", "마이프로틴"],
            "products": ["닭가슴살 팩", "닭가슴살 큐브", "후추맛 닭가슴살", "닭가슴살 갈릭", "초코맛 파우더", "프로틴 파우더", "프로틴", "부스터"]
        },
        6: {  # 유아
            "brands": ["헤겐", "팝앤고", "타이니 트윙클", "베이비멜"],
            "products": ["젖병", "쪽쪽이", "턱받이", "대용량 젖병", "기저귀", "물티슈", "로션"]
        }
    }
    extras = ["10개", "20개", "30개", "5개", "24개"]
    # ======================================

    # 1. 아이템별 카테고리 우선 할당 (백엔드 규칙: 1~6)
    item_category_ids = np.random.choice(list(category_data.keys()), size=n_items)

    # 2. 할당된 카테고리에 따라 제목 생성
    item_titles = []
    for category_id in item_category_ids:
        brand = np.random.choice(category_data[category_id]["brands"])
        product = np.random.choice(category_data[category_id]["products"])
        extra = np.random.choice(extras)
        title = f"{brand} {product} {extra}"
        item_titles.append(title)

    # 3. item_meta 데이터프레임 생성
    item_meta = pd.DataFrame({
        "item_id": item_ids,
        "category_idx": item_category_ids,  # 백엔드에서 제공하는 int 값 (1~6)
        "price": np.random.randint(500, 250000, size=n_items),
        "title": item_titles
    })

    # 4. 이벤트 데이터 생성
    start_time = pd.Timestamp("2025-10-01")
    timestamps = [start_time + pd.Timedelta(seconds=int(x)) for x in np.random.randint(0, 86400 * 30, size=n_events)]
    df = pd.DataFrame({
        "user_id": np.random.choice(user_ids, size=n_events),
        "item_id": np.random.choice(item_ids, size=n_events),
        "action": np.random.choice(["view", "like", "enter", "purchase"], p=[0.7, 0.15, 0.09, 0.06], size=n_events),
        # view: 게시물 보기, like: 좋아요, enter: 구매 진입, purchase: 실제 구매
        "timestamp": timestamps
    })
    df = df.merge(item_meta[["item_id", "category_idx", "price", "title"]], on="item_id", how="left")
    
    return df, item_meta, user_ids, item_ids


def process_and_save(df, item_meta, user_ids, item_ids, output_dir="output"):
    """
    데이터를 전처리하고 학습에 필요한 형태로 저장하는 함수
    - 네거티브 샘플링 제거 (모델 학습 시 수행)
    - Train/Val/Test 스키마 통일
    
    Args:
        df: 이벤트 데이터프레임
        item_meta: 아이템 메타데이터
        user_ids: 사용자 ID 리스트
        item_ids: 아이템 ID 리스트
        output_dir: 출력 디렉토리 경로
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) 암시적 피드백을 위한 레이블과 가중치 매핑
    positive_actions = {"purchase": 1, "enter": 1, "like": 1, "view": 1}
    df["label"] = df["action"].map(positive_actions).fillna(0).astype(int)
    df["weight"] = df["action"].map({"purchase": 5, "enter": 3, "like": 2, "view": 1}).fillna(1)

    # 2) 범주형 변수 인코딩 (user, item만 - category_idx는 이미 백엔드 규칙에 따라 1~6)
    le_user = LabelEncoder().fit(df["user_id"].unique())
    le_item = LabelEncoder().fit(df["item_id"].unique())
    df["user_idx"] = le_user.transform(df["user_id"])
    df["item_idx"] = le_item.transform(df["item_id"])
    
    # category_idx는 이미 백엔드에서 제공한 값 (1~6)이므로 그대로 사용
    # df["category_idx"]는 이미 존재함

    # 3) 수치형 변수 정규화
    scaler_price = StandardScaler().fit(item_meta[["price"]])
    df["price_norm"] = scaler_price.transform(df[["price"]])

    # 4) 시간 기반 데이터 분할
    df = df.sort_values("timestamp").reset_index(drop=True)
    train_cut = df["timestamp"].quantile(0.70)
    val_cut = df["timestamp"].quantile(0.85)
    train_df = df[df["timestamp"] <= train_cut].copy()
    val_df = df[(df["timestamp"] > train_cut) & (df["timestamp"] <= val_cut)].copy()
    test_df = df[df["timestamp"] > val_cut].copy()

    # 콜드스타트 문제 방지: 상호작용이 적은 사용자 제외
    min_interactions = 3
    user_counts = train_df["user_id"].value_counts()
    active_users = user_counts[user_counts >= min_interactions].index
    train_df = train_df[train_df["user_id"].isin(active_users)].copy()

    # 5) 학습에 필요한 컬럼만 선택 (네거티브 샘플링 제거)
    # 네거티브 샘플링은 ImplicitMatrixFactorization 학습 시 자동으로 수행됨
    required_cols = ["user_idx", "item_idx", "label", "weight", "price_norm", "category_idx"]
    
    train_final = train_df[required_cols].copy()
    val_final = val_df[required_cols].copy()
    test_final = test_df[required_cols].copy()

    # 6) 결과 저장 (모든 데이터셋이 동일한 스키마)
    train_final.to_csv(os.path.join(output_dir, "user_item_train.csv"), index=False)
    val_final.to_csv(os.path.join(output_dir, "user_item_val.csv"), index=False)
    test_final.to_csv(os.path.join(output_dir, "user_item_test.csv"), index=False)

    # 7) 인코더 메타데이터 저장
    enc_meta = {
        "user_classes": le_user.classes_.tolist(), 
        "item_classes": le_item.classes_.tolist(),
        "category_mapping": CATEGORY_MAPPING,  # 백엔드 카테고리 매핑 정보
        "price_mean": float(scaler_price.mean_[0]),
        "price_var": float(scaler_price.var_[0])
    }
    with open(os.path.join(output_dir, "encoders.json"), "w", encoding='utf-8') as f:
        json.dump(enc_meta, f, indent=2, ensure_ascii=False)

    # 8) 통계 출력
    print("\n" + "=" * 70)
    print("데이터 생성 완료")
    print("=" * 70)
    print(f"저장 경로: {output_dir}")
    print(f"\n📊 데이터 크기:")
    print(f"  - Train: {len(train_final):,}개 (Positive only)")
    print(f"  - Val: {len(val_final):,}개")
    print(f"  - Test: {len(test_final):,}개")
    print(f"\n📋 스키마: {list(train_final.columns)}")
    print(f"\n🏷️  카테고리 매핑 (백엔드 규칙):")
    for cat_id, cat_name in CATEGORY_MAPPING.items():
        cat_count = train_final[train_final['category_idx'] == cat_id].shape[0]
        print(f"  - {cat_id}: {cat_name} ({cat_count:,}개)")
    print(f"\n⚠️  주의: 네거티브 샘플링은 ImplicitMatrixFactorization.fit() 시 자동으로 수행됩니다.")
    print("=" * 70)


def main():
    """메인 함수"""
    print("=" * 70)
    print("Splitty 추천 시스템 데이터 생성 시작")
    print("=" * 70)
    
    # 데이터 생성
    print("\n[1/2] 가상 데이터 생성 중...")
    df, item_meta, user_ids, item_ids = make_synthetic_data(
        n_users=200,
        n_items=500,
        n_events=7000,
        seed=42
    )
    
    # 전처리 및 저장
    print("\n[2/2] 데이터 전처리 및 저장 중...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(os.path.dirname(current_dir), "splitty_recommendation_data_1")
    process_and_save(df, item_meta, user_ids, item_ids, output_dir=output_dir)
    
    # 검증
    print("\n✅ 생성된 데이터 샘플:")
    train = pd.read_csv(os.path.join(output_dir, "user_item_train.csv"))
    print(train.head(3))


if __name__ == "__main__":
    main()
