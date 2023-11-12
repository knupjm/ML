# 기계학습 - 10일차

### 차원의 저주

- `차원의 저주` : 차원이 증가할수록 데이터 공간이 점점 희소해지는 현상.
- `차원 축소` : 차원의 저주를 극복하기 위한 방법.
  - `투영` : 데이터를 저차원 공간으로 투영하는 방법. ex) PCA
  - `매니폴드 학습` : 데이터가 존재하는 저차원 공간을 모델링하는 방법. ex) LLE

### 주성분 찾기

#### 주성분 축 찾기

`첫 번째 주성분` : 훈련 세트에서 분산을 최대한 보존하는 축.  
`두 번째 주성분` : 첫 번째 주성분과 수직을 이루면서 분산을 최대한 보존하는 축  
`세 번째 주성분` : 첫 번째, 두 번째 주성분과 수직을 이루면서 분산을 최대한 보존하는 축

#### 훈련 세트의 주성분 찾기

SVD 사용 예제

```python
X_centered = X - X.mean(axis=0) # 평균을 0으로
U, s, Vt = np.linalg.svd(X_centered) # SVD
c1 = Vt.T[:, 0] # 첫번째 주성분
c2 = Vt.T[:, 1] # 두번째 주성분
```

#### d차원으로 투영하기

```python
W2 = Vt.T[:, :2] # 첫 두개의 주성분
X2D = X_centered.dot(W2) # 투영
```

### 사이킷런(PCA) 사용하기

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2) # 주성분 2개
X2D = pca.fit_transform(X) # 투영
```

#### 적절한 차원 수 선택하기

```python
pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_) # 누적 분산
d = np.argmax(cumsum >= 0.95) + 1 # 분산이 95% 이상이 되는 차원 수
```

분산이 **95% 이상**이 되는 차원 수를 선택하면  
대부분의 분산을 유지하면서도 데이터셋을 2D로 투영할 수 있음.

#### 압축을 위한 PCA

```python
pca = PCA(n_components=154)
X_reduced = pca.fit_transform(X_train) # 154차원으로 압축
X_recovered = pca.inverse_transform(X_reduced) # 784차원으로 복원
```

#### 랜덤 PCA

```python
rnd_pca = PCA(n_components=154, svd_solver="randomized" randon_state=42)
X_reduced = rnd_pca.fit_transform(X_train)
```

랜덤 PCA라 부르는 확률적 알고리즘을 사용해  
처음 d개의 주성분에 대한 **근삿값**을 빠르게 찾음.

#### 점진적 PCA

```python
from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    print(".", end="")
    inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X_train)
```

온라인 학습에 사용할 수 있는 `점진적 PCA` 구현.

#### 커널 PCA

```python
from sklearn.decomposition import KernelPCA

rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)
```

투영된 후에 샘플의 군집을 유지하거나  
꼬인 매니폴드에 가까운 데이터 셋을 펼칠 때 유용

#### LLE

```python
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_reduced = lle.fit_transform(X)
```

`LLE`는 매니폴드 학습에 사용하는 또 다른 비선형 차원 축소 기법.

### 다양한 차원 축소 기법

1. `다차원 스케일링(MDS)` : 샘플 간의 거리를 보존하면서 차원을 축소.
2. `등각 사상(Isomap)` : 각 샘플을 가장 가까운 이웃과 연결하는 식으로 그래프를 만든 후에 샘플 간의 지오데식 거리(Geodesic distance)를 유지하며 차원을 축소.
3. `t-SNE` : 비슷한 샘플은 가깝게 두고, 비슷하지 않은 샘플은 멀리 떨어지도록 차원을 축소.
4. `선형 판별 분석(LDA)` : 분류 알고리즘. 클래스 사이를 가장 잘 구분하는 축을 학습.
5. `NMF` : 특성이 양수인 데이터셋에 적용. 두 개의 행렬 W와 H로 분해.
6. `랜덤 투영` : 무작위 선형 투영을 사용해 데이터를 저차원 공간으로 투영. 랜덤 투영은 차원 축소에 아주 효율적이지만, PCA나 점진적 PCA, 랜덤 PCA 등을 사용하는 것이 더 유용.
