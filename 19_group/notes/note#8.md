### 비지도 학습

- `비지도 학습` : 레이블이 없는 데이터를 다루는 머신러닝
- `군집` : 비지도 학습의 대표적인 예로 비슷한 데이터끼리 **그룹**으로 묶는 작업
- `이상치 탐지` : 데이터셋에서 **비정상**적인 데이터를 감지
- `밀도 추정` : 데이터셋에서 **확률**적 모델을 추정

### K-평균

- K-평균(로이드-포지 알고리즘)은 가장 간단하고 널리 사용하는 군집 알고리즘
- 클러스터 개수를 미리 지정해야 함
- 군집 크기의 차이가 크면 잘 작동하지 않음
- `결정 경계` : 군집을 구분하는 경계
- `하드 군집` : 각 샘플에 대해 가장 가까운 클러스터 선택
- `소프트 군집` : 클러스터마다 샘플에 점수 부여

```python
from sklearn.cluster import KMeans

k = 5 # 군집 개수
kmeans = KMeans(n_clusters=km, random_state=42) # 모델 생성
y_pred = kmeans.fit_predict(X) # 모델 학습
```

### K-평균 알고리즘의 절차

1. 무작위로 k개의 샘플을 선택해 센트로이드로 지정
2. 각 샘플을 가장 가까운 센트로이드에 할당
3. 센트로이드에 할당된 샘플의 평균으로 센트로이드 이동
4. 2~3번을 수렴할 때까지 반복

### K-평균 최적의 클러스터 개수 찾기

- 관성이 더 이상 크게 줄어들지 않는 지점의 클러스터 개수 선택
- `실루엣 점수` : 클러스터가 얼마나 잘 구분되어 있는지 평가하는 지표
  - +1에 가까운 값 : 자신의 클러스터 안에 잘 속해 있고, 다른 클러스터와는 멀리 떨어져 있다는 뜻
  - 0에 가까운 값 : 클러스터 경계에 위치
  - -1에 가까운 값 : 이 샘플이 잘못된 클러스터에 할당되었다는 뜻
- `실루엣 다이어그램` : 클러스터 개수를 시각화하는 방법(칼 모양의 그래프)
- `빨간 파선` : 대부분의 칼이 빨간 파선보다 길어야 함

### 군집을 사용한 이미지 분할

- `이미지 분할` : 이미지를 세그먼트 여러 개로 분할하는 작업
- `시맨틱 분할` : 동일한 종류의 물체에 속한 모든 픽셀을 같은 세그먼트로 할당
- `색상 분할` : 비슷한 색상을 가진 픽셀을 같은 세그먼트로 할당

### DBSCAN

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.05, min_samples=5) # 모델 생성
dbscan.fit(X) # 모델 학습
```

- `DBSCAN` : 밀집된 지역을 구분하는 군집 알고리즘
  - `eps` : 샘플 사이의 최대 거리
  - `min_samples` : 샘플이 속한 세트의 최소 샘플 개수
- `predict()` 함수를 지원하지 않고 `fit_predict()` 함수 제공
- `fit_predict()` : 노이즈 포함한 클러스터 레이블을 반환

#### GMM(가우시안 혼합 모델)

```python
from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=3, n_init=10, random_state=42) # 모델 생성
gm.fit(X) # 모델 학습

gm.weights_ # 각 클러스터의 상대적인 가중치
gm.means_ # 각 클러스터의 평균
gm.covariances_ # 각 클러스터의 공분산 행렬
```

- `GMM` : 샘플이 어떤 통계적인 분포를 따르고 있다고 가정하는 **확률적 모델**
  - `convergence_iter` : 공분산 규제
    - `full` : 제한 없음
    - `spherical` : 모든 클러스터가 원형이지만 지름은 다를 수 있음
    - `diag` : 어떤 타원형도 가능, 타원의 축이 좌표축과 평행하다고 가정
    - `tied` : 모든 군집의 동일 모양, 동일 크기, 동일 방향을 갖는다고 가정

> GMM은 타원형 클러스터에서 잘 작동하지만, 다른 모양의 클러스터에서는 잘 작동하지 않음

### GMM을 사용한 이상치 탐지

```python
densities = gm.score_samples(X) # 샘플의 밀도 계산
density_threshold = np.percentile(densities, 4) # 4% 밀도 계산
anomalies = X[densities < density_threshold] # 이상치 추출
```

### 베이즈 가우시안 혼합 모델

```python
from sklearn.mixture import BayesianGaussianMixture

bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42) # 모델 생성
bgm.fit(X)
```

- `BGMM` : 최적의 군집수를 자동으로 찾아줌
