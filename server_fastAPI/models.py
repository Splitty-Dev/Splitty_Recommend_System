"""
Hybrid Recommendation System
Matrix Factorization + Two-Tower Model

구조:
1. Matrix Factorization → Top-K Candidates (K=250)
2. Two-Tower Model → Top-N Personalized Ranking (N=50)
3. Final Output → 사용자에게 추천
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import pickle
from typing import List, Tuple, Dict, Optional
import warnings
import random
warnings.filterwarnings('ignore')


class ImplicitMatrixFactorization:
    """
    네거티브 샘플링을 포함한 Implicit Feedback 기반 Matrix Factorization
    ALS (Alternating Least Squares) 대신 SGD를 사용하여 positive/negative 샘플링 지원
    """
    
    def __init__(self, n_factors=50, learning_rate=0.01, regularization=0.01, 
                 n_epochs=100, negative_samples=5, random_state=42):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.negative_samples = negative_samples
        self.random_state = random_state
        
        self.user_factors = None
        self.item_factors = None
        self.user_encoder = {}
        self.item_encoder = {}
        self.user_decoder = {}
        self.item_decoder = {}
        
    def _generate_negative_samples(self, df: pd.DataFrame, n_negative: int = 5):
        """
        각 positive 상호작용에 대해 네거티브 샘플 생성
        """
        print(f"네거티브 샘플링 시작 (비율: 1:{n_negative})")
        
        # 사용자별 상호작용한 아이템 세트
        user_items = df.groupby('user_idx')['item_idx'].apply(set).to_dict()
        all_items = set(df['item_idx'].unique())
        
        negative_samples = []
        
        for user_idx, interacted_items in user_items.items():
            # 상호작용하지 않은 아이템들
            uninteracted_items = all_items - interacted_items
            
            # 각 positive 상호작용에 대해 네거티브 샘플 생성
            n_positives = len(interacted_items)
            n_negatives_for_user = min(n_negative * n_positives, len(uninteracted_items))
            
            if n_negatives_for_user > 0:
                sampled_negative_items = random.sample(list(uninteracted_items), n_negatives_for_user)
                
                for item_idx in sampled_negative_items:
                    negative_samples.append({
                        'user_idx': user_idx,
                        'item_idx': item_idx,
                        'weight': 0,  # 네거티브 샘플
                        'label': 0
                    })
        
        print(f"네거티브 샘플 생성 완료: {len(negative_samples)}개")
        return pd.DataFrame(negative_samples)
    
    def fit(self, df: pd.DataFrame, use_negative_sampling: bool = True):
        """
        Implicit Feedback MF 모델 학습
        """
        print("Implicit Matrix Factorization 학습 시작...")
        
        # 인코더 준비
        unique_users = df['user_idx'].unique()
        unique_items = df['item_idx'].unique()
        
        self.user_encoder = {user_idx: idx for idx, user_idx in enumerate(unique_users)}
        self.item_encoder = {item_idx: idx for idx, item_idx in enumerate(unique_items)}
        self.user_decoder = {idx: user_idx for user_idx, idx in self.user_encoder.items()}
        self.item_decoder = {idx: item_idx for item_idx, idx in self.item_encoder.items()}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        # 학습 데이터 준비
        train_data = df.copy()
        
        # 네거티브 샘플링 적용
        if use_negative_sampling:
            negative_samples = self._generate_negative_samples(df, self.negative_samples)
            train_data = pd.concat([train_data, negative_samples], ignore_index=True)
        
        # 인덱스 변환
        user_indices = [self.user_encoder[uid] for uid in train_data['user_idx']]
        item_indices = [self.item_encoder[iid] for iid in train_data['item_idx']]
        ratings = train_data['weight'].values
        
        # 팩터 매트릭스 초기화
        np.random.seed(self.random_state)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # SGD 학습
        print(f"SGD 학습 시작 (epochs: {self.n_epochs})")
        
        for epoch in range(self.n_epochs):
            total_loss = 0
            
            # 랜덤 순서로 학습
            indices = list(range(len(user_indices)))
            random.shuffle(indices)
            
            for i in indices:
                user_idx = user_indices[i]
                item_idx = item_indices[i]
                rating = ratings[i]
                
                # 예측값 계산
                prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
                
                # 오차 계산
                error = rating - prediction
                total_loss += error ** 2
                
                # 그래디언트 업데이트
                user_factor = self.user_factors[user_idx].copy()
                item_factor = self.item_factors[item_idx].copy()
                
                self.user_factors[user_idx] += self.learning_rate * (
                    error * item_factor - self.regularization * user_factor
                )
                self.item_factors[item_idx] += self.learning_rate * (
                    error * user_factor - self.regularization * item_factor
                )
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(indices)
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {avg_loss:.4f}")
        
        print("Implicit Matrix Factorization 학습 완료!")
        print(f"사용자 수: {n_users}, 아이템 수: {n_items}, Factors: {self.n_factors}")
        
    def get_top_k_candidates(self, user_id: str, k: int = 250, 
                           exclude_seen: bool = True, 
                           interaction_df: pd.DataFrame = None,
                           category_filter = None,
                           available_items: List[int] = None) -> List[Tuple[str, float]]:
        """
        특정 사용자에게 Top-K 후보 아이템 반환
        
        Args:
            user_id: 사용자 ID
            k: 반환할 후보 개수
            exclude_seen: 이미 상호작용한 아이템 제외 여부
            interaction_df: 상호작용 데이터프레임
            category_filter: 특정 카테고리 인덱스로 필터링 (옵션, int 또는 None)
        """
        if user_id not in self.user_encoder:
            # 새로운 사용자인 경우 인기 아이템 반환
            return self._get_popular_items(k, interaction_df, category_filter)
        
        user_idx = self.user_encoder[user_id]
        user_vector = self.user_factors[user_idx]
        
        # 모든 아이템에 대한 예측 점수 계산
        scores = np.dot(user_vector, self.item_factors.T)
        
        # 카테고리 필터링
        if category_filter is not None and interaction_df is not None:
            # 해당 카테고리가 아닌 아이템들의 점수를 -inf로 설정
            category_items = set(interaction_df[interaction_df['category_idx'] == category_filter]['item_idx'].values)
            for item_id in self.item_encoder:
                if item_id not in category_items:
                    item_idx = self.item_encoder[item_id]
                    scores[item_idx] = -np.inf
        
        # available_items 필터링 (거리 기반으로 제한된 아이템만 고려)
        if available_items is not None:
            available_items_set = set(available_items)
            for item_id in self.item_encoder:
                if item_id not in available_items_set:
                    item_idx = self.item_encoder[item_id]
                    scores[item_idx] = -np.inf
        
        # 이미 상호작용한 아이템 제외
        if exclude_seen and interaction_df is not None:
            seen_items = set(interaction_df[interaction_df['user_idx'] == user_id]['item_idx'].values)
            for item_id in seen_items:
                if item_id in self.item_encoder:
                    item_idx = self.item_encoder[item_id]
                    scores[item_idx] = -np.inf
        
        # Top-K 선택
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        candidates = []
        for idx in top_k_indices:
            if scores[idx] > -np.inf:  # 유효한 점수인 경우만
                item_id = self.item_decoder[idx]
                candidates.append((item_id, float(scores[idx])))
        
        return candidates[:k]
    
    def _get_popular_items(self, k: int, interaction_df: pd.DataFrame, category_filter=None) -> List[Tuple[str, float]]:
        """
        신규 사용자를 위한 인기 아이템 반환
        """
        if interaction_df is None:
            return []
        
        # 카테고리 필터링 적용
        df = interaction_df
        if category_filter is not None:
            df = df[df["category_idx"] == category_filter]

        # 가중치 기반 인기도 계산
        popularity = df.groupby("item_idx")["weight"].sum().sort_values(ascending=False)        
        candidates = []
        for item_id, score in popularity.head(k).items():
            candidates.append((item_id, float(score)))
            
        return candidates
    
    def save_model(self, filepath: str):
        """모델 저장"""
        model_data = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'user_decoder': self.user_decoder,
            'item_decoder': self.item_decoder,
            'n_factors': self.n_factors,
            'learning_rate': self.learning_rate,
            'regularization': self.regularization,
            'negative_samples': self.negative_samples
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Implicit MF 모델이 {filepath}에 저장되었습니다.")
    
    def load_model(self, filepath: str):
        """모델 로드"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.user_factors = model_data['user_factors']
        self.item_factors = model_data['item_factors']
        self.user_encoder = model_data['user_encoder']
        self.item_encoder = model_data['item_encoder']
        self.user_decoder = model_data['user_decoder']
        self.item_decoder = model_data['item_decoder']
        self.n_factors = model_data['n_factors']
        self.learning_rate = model_data.get('learning_rate', 0.01)
        self.regularization = model_data.get('regularization', 0.01)
        self.negative_samples = model_data.get('negative_samples', 5)
        
        print(f"Implicit MF 모델이 {filepath}에서 로드되었습니다.")


class MatrixFactorization:
    """
    Matrix Factorization을 사용한 협업 필터링 모델
    사용자-아이템 상호작용 매트릭스를 분해하여 Top-K 후보를 생성
    """
    
    def __init__(self, n_factors=50, random_state=42):
        self.n_factors = n_factors
        self.random_state = random_state
        self.model = None
        self.user_factors = None
        self.item_factors = None
        self.user_encoder = {}
        self.item_encoder = {}
        self.user_decoder = {}
        self.item_decoder = {}
        
    def prepare_data(self, df: pd.DataFrame) -> csr_matrix:
        """
        사용자-아이템 상호작용 매트릭스 생성
        """
        # 사용자와 아이템 인코딩
        unique_users = df['user_idx'].unique()
        unique_items = df['item_idx'].unique()
        
        self.user_encoder = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_encoder = {item_id: idx for idx, item_id in enumerate(unique_items)}
        self.user_decoder = {idx: user_id for user_id, idx in self.user_encoder.items()}
        self.item_decoder = {idx: item_id for item_id, idx in self.item_encoder.items()}
        
        # 상호작용 매트릭스 생성 (가중치 적용)
        rows = [self.user_encoder[user_id] for user_id in df['user_idx']]
        cols = [self.item_encoder[item_id] for item_id in df['item_idx']]
        data = df['weight'].values
        
        interaction_matrix = csr_matrix(
            (data, (rows, cols)), 
            shape=(len(unique_users), len(unique_items))
        )
        
        return interaction_matrix
    
    def fit(self, df: pd.DataFrame):
        """
        Matrix Factorization 모델 학습
        """
        print("Matrix Factorization 모델 학습 시작...")
        
        # 상호작용 매트릭스 생성
        interaction_matrix = self.prepare_data(df)
        
        # NMF를 사용한 행렬 분해
        self.model = NMF(
            n_components=self.n_factors,
            init='random',
            random_state=self.random_state,
            max_iter=200
        )
        
        # 학습 실행
        self.user_factors = self.model.fit_transform(interaction_matrix)
        self.item_factors = self.model.components_.T
        
        print(f"Matrix Factorization 학습 완료!")
        print(f"사용자 수: {len(self.user_encoder)}, 아이템 수: {len(self.item_encoder)}")
        print(f"Factors: {self.n_factors}")
        
    def get_top_k_candidates(self, user_id: str, k: int = 250, 
                           exclude_seen: bool = True, 
                           interaction_df: pd.DataFrame = None,
                           available_items: List[int] = None) -> List[Tuple[str, float]]:
        """
        특정 사용자에게 Top-K 후보 아이템 반환
        """
        if user_id not in self.user_encoder:
            # 새로운 사용자인 경우 인기 아이템 반환
            return self._get_popular_items(k, interaction_df)
        
        user_idx = self.user_encoder[user_id]
        user_vector = self.user_factors[user_idx]
        
        # 모든 아이템에 대한 예측 점수 계산
        scores = np.dot(user_vector, self.item_factors.T)
        
        # available_items 필터링 (거리 기반으로 제한된 아이템만 고려)
        if available_items is not None:
            available_items_set = set(available_items)
            for item_id in self.item_encoder:
                if item_id not in available_items_set:
                    item_idx = self.item_encoder[item_id]
                    scores[item_idx] = -np.inf
        
        # 이미 상호작용한 아이템 제외
        if exclude_seen and interaction_df is not None:
            seen_items = set(interaction_df[interaction_df['user_idx'] == user_id]['item_idx'].values)
            for item_id in seen_items:
                if item_id in self.item_encoder:
                    item_idx = self.item_encoder[item_id]
                    scores[item_idx] = -np.inf
        
        # Top-K 선택
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        candidates = []
        for idx in top_k_indices:
            if scores[idx] > -np.inf:  # 유효한 점수인 경우만
                item_id = self.item_decoder[idx]
                candidates.append((item_id, float(scores[idx])))
        
        return candidates[:k]
    
    def _get_popular_items(self, k: int, interaction_df: pd.DataFrame, category_filter=None) -> List[Tuple[str, float]]:
        """
        신규 사용자를 위한 인기 아이템 반환
        """
        if interaction_df is None:
            return []
        
        # 카테고리 필터링 적용
        df = interaction_df
        if category_filter is not None:
            df = df[df["category_idx"] == category_filter]

        # 가중치 기반 인기도 계산
        popularity = df.groupby("item_idx")["weight"].sum().sort_values(ascending=False)        
        candidates = []
        for item_id, score in popularity.head(k).items():
            candidates.append((item_id, float(score)))
            
        return candidates
    
    def save_model(self, filepath: str):
        """모델 저장"""
        model_data = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'user_decoder': self.user_decoder,
            'item_decoder': self.item_decoder,
            'n_factors': self.n_factors
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Matrix Factorization 모델이 {filepath}에 저장되었습니다.")
    
    def load_model(self, filepath: str):
        """모델 로드"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.user_factors = model_data['user_factors']
        self.item_factors = model_data['item_factors']
        self.user_encoder = model_data['user_encoder']
        self.item_encoder = model_data['item_encoder']
        self.user_decoder = model_data['user_decoder']
        self.item_decoder = model_data['item_decoder']
        self.n_factors = model_data['n_factors']
        
        print(f"Matrix Factorization 모델이 {filepath}에서 로드되었습니다.")


class TwoTowerModel(nn.Module):
    """
    Two-Tower Neural Network for Personalized Ranking
    사용자 타워와 아이템 타워로 구성된 딥러닝 모델 (제목 임베딩 포함)
    """
    
    def __init__(self, n_users: int, n_items: int, n_categories: int, 
                 embedding_dim: int = 64, hidden_dims: List[int] = [128, 64],
                 title_embedding_dim: int = 768):
        super(TwoTowerModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.title_embedding_dim = title_embedding_dim
        
        # User Tower
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        
        # Item Tower  
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.category_embedding = nn.Embedding(n_categories, embedding_dim // 2)
        
        # 제목 임베딩 차원 축소 (768 → 64)
        self.title_projection = nn.Sequential(
            nn.Linear(title_embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # User Tower의 FC layers
        user_layers = []
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            user_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        self.user_tower = nn.Sequential(*user_layers)
        
        # Item Tower의 FC layers (제목 임베딩 포함)
        item_layers = []
        input_dim = embedding_dim + (embedding_dim // 2) + embedding_dim + 1  # item_emb + category_emb + title_emb + price_norm
        for hidden_dim in hidden_dims:
            item_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        self.item_tower = nn.Sequential(*item_layers)
        
        # Final output layer
        self.output_layer = nn.Linear(1, 1)
        
    def forward(self, user_ids, item_ids, category_ids, price_norm, title_embeddings=None):
        # User Tower
        user_emb = self.user_embedding(user_ids)
        user_features = self.user_tower(user_emb)
        
        # Item Tower
        item_emb = self.item_embedding(item_ids)
        category_emb = self.category_embedding(category_ids)
        price_norm = price_norm.unsqueeze(1) if price_norm.dim() == 1 else price_norm
        
        # 제목 임베딩 처리
        if title_embeddings is not None:
            title_emb = self.title_projection(title_embeddings)
        else:
            # 제목 임베딩이 없는 경우 영벡터 사용
            title_emb = torch.zeros(item_emb.shape[0], self.embedding_dim, device=item_emb.device)
        
        item_input = torch.cat([item_emb, category_emb, title_emb, price_norm], dim=1)
        item_features = self.item_tower(item_input)
        
        # Interaction (dot product)
        interaction = torch.sum(user_features * item_features, dim=1)
        output = self.output_layer(interaction.unsqueeze(1))
        
        return torch.sigmoid(output.squeeze())


class TwoTowerTrainer:
    """
    Two-Tower 모델 학습 및 추론을 위한 클래스 (제목 임베딩 포함)
    """
    
    def __init__(self, model: TwoTowerModel, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.user_encoder = {}
        self.item_encoder = {}
        self.category_encoder = {}
        self.title_embeddings = None
        
    def load_title_embeddings(self, data_dir: str):
        """제목 임베딩 로드"""
        try:
            import os
            embedding_path = os.path.join(data_dir, "item_title_embeddings.npz")
            if os.path.exists(embedding_path):
                embeddings_data = np.load(embedding_path)
                self.title_embeddings = embeddings_data['embeddings']
                print(f"제목 임베딩 로드 완료: {self.title_embeddings.shape}")
                return True
            else:
                print(f"제목 임베딩 파일을 찾을 수 없습니다: {embedding_path}")
                return False
        except Exception as e:
            print(f"제목 임베딩 로드 실패: {str(e)}")
            return False
        
    def prepare_encoders(self, df: pd.DataFrame):
        """인코더 준비"""
        unique_users = df['user_idx'].unique()
        unique_items = df['item_idx'].unique()
        unique_categories = df['category_idx'].unique()
        
        self.user_encoder = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_encoder = {item_id: idx for idx, item_id in enumerate(unique_items)}
        self.category_encoder = {cat: idx for idx, cat in enumerate(unique_categories)}
    
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None, 
            epochs: int = 50, batch_size: int = 1024, lr: float = 0.001,
            data_dir: str = None):
        """모델 학습 (제목 임베딩 포함)"""
        print("Two-Tower 모델 학습 시작 (제목 임베딩 포함)...")
        
        # 제목 임베딩 로드
        if data_dir:
            self.load_title_embeddings(data_dir)
        
        self.prepare_encoders(train_df)
        
        # 데이터 준비
        train_dataset = self._prepare_dataset(train_df)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                user_ids, item_ids, category_ids, price_norm, labels, title_embs = batch
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                category_ids = category_ids.to(self.device)
                price_norm = price_norm.to(self.device)
                labels = labels.to(self.device)
                title_embs = title_embs.to(self.device) if title_embs is not None else None
                
                optimizer.zero_grad()
                outputs = self.model(user_ids, item_ids, category_ids, price_norm, title_embs)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        print("Two-Tower 모델 학습 완료!")
    
    def _prepare_dataset(self, df: pd.DataFrame):
        """PyTorch Dataset 준비 (제목 임베딩 포함)"""
        user_ids = [self.user_encoder[uid] for uid in df['user_idx']]
        item_ids = [self.item_encoder[iid] for iid in df['item_idx']]
        category_ids = [self.category_encoder[cat] for cat in df['category_idx']]
        price_norm = df['price_norm'].values
        labels = df['label'].values
        
        # 제목 임베딩 준비
        title_embeddings = None
        if self.title_embeddings is not None:
            # 아이템 ID를 인덱스로 매핑하여 제목 임베딩 추출
            title_embs = []
            for item_id in df['item_idx']:
                item_idx = self.item_encoder[item_id]
                if item_idx < len(self.title_embeddings):
                    title_embs.append(self.title_embeddings[item_idx])
                else:
                    # 임베딩이 없는 경우 영벡터 사용
                    title_embs.append(np.zeros(self.title_embeddings.shape[1]))
            title_embeddings = torch.FloatTensor(np.array(title_embs))
        else:
            # 제목 임베딩이 없는 경우 None 사용
            title_embeddings = None
        
        if title_embeddings is not None:
            dataset = torch.utils.data.TensorDataset(
                torch.LongTensor(user_ids),
                torch.LongTensor(item_ids), 
                torch.LongTensor(category_ids),
                torch.FloatTensor(price_norm),
                torch.LongTensor(labels),
                title_embeddings
            )
        else:
            # 제목 임베딩 없이 기존 방식 유지
            dataset = torch.utils.data.TensorDataset(
                torch.LongTensor(user_ids),
                torch.LongTensor(item_ids), 
                torch.LongTensor(category_ids),
                torch.FloatTensor(price_norm),
                torch.LongTensor(labels),
                torch.zeros(len(user_ids), 768)  # 더미 제목 임베딩
            )
        return dataset
    
    def predict_scores(self, user_id: str, candidate_items: List[str], 
                      item_meta: pd.DataFrame) -> List[Tuple[str, float]]:
        """후보 아이템들에 대한 예측 점수 계산 (제목 임베딩 포함)"""
        if user_id not in self.user_encoder:
            # 새 사용자인 경우 랜덤 점수 (또는 다른 전략)
            return [(item_id, np.random.random()) for item_id in candidate_items]
        
        self.model.eval()
        scored_items = []
        
        with torch.no_grad():
            user_idx = self.user_encoder[user_id]
            
            for item_id in candidate_items:
                if item_id not in self.item_encoder:
                    continue
                    
                item_idx = self.item_encoder[item_id]
                
                # 아이템 메타데이터 가져오기
                item_info = item_meta[item_meta['item_id'] == item_id]
                if item_info.empty:
                    continue
                
                category = item_info.iloc[0]['category']
                price_norm = item_info.iloc[0]['price_norm']
                
                if category not in self.category_encoder:
                    continue
                
                category_idx = self.category_encoder[category]
                
                # 제목 임베딩 가져오기
                title_emb = None
                if self.title_embeddings is not None and item_idx < len(self.title_embeddings):
                    title_emb = torch.FloatTensor(self.title_embeddings[item_idx]).unsqueeze(0).to(self.device)
                
                # 텐서 생성
                user_tensor = torch.LongTensor([user_idx]).to(self.device)
                item_tensor = torch.LongTensor([item_idx]).to(self.device)
                category_tensor = torch.LongTensor([category_idx]).to(self.device)
                price_tensor = torch.FloatTensor([price_norm]).to(self.device)
                
                # 예측
                score = self.model(user_tensor, item_tensor, category_tensor, price_tensor, title_emb)
                scored_items.append((item_id, float(score.cpu().numpy())))
        
        # 점수 기준 정렬
        scored_items.sort(key=lambda x: x[1], reverse=True)
        return scored_items
    
    def save_model(self, filepath: str):
        """모델 저장"""
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'category_encoder': self.category_encoder,
            'model_config': {
                'n_users': len(self.user_encoder),
                'n_items': len(self.item_encoder),
                'n_categories': len(self.category_encoder),
                'embedding_dim': self.model.embedding_dim,
                'hidden_dims': self.model.hidden_dims
            }
        }
        
        torch.save(model_data, filepath)
        print(f"Two-Tower 모델이 {filepath}에 저장되었습니다.")
    
    def load_model(self, filepath: str):
        """모델 로드"""
        model_data = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(model_data['model_state_dict'])
        self.user_encoder = model_data['user_encoder']
        self.item_encoder = model_data['item_encoder']
        self.category_encoder = model_data['category_encoder']
        
        print(f"Two-Tower 모델이 {filepath}에서 로드되었습니다.")