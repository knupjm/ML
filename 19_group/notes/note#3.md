# 기계학습 - 4일차

## 정리노트

### 선형 회귀

- `선형 회귀 모델 예측` : 입력 특성의 가중치 합과 편향이라는 상수를 더해 예측
- `정규방정식` : 비용 함수를 최소화하는 θ값을 찾기위한 해석적인 방법

### 선형 회귀 예제 분석

```python
# 선형 회귀의 정규 방정식을 사용하여 최적의 매개변수 (theta) 계산
# .dot() : 행렬 곱셈
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
```

```python
lin_reg = LinearRegression()  # 선형 회귀 모델 객체 생성
lin_reg.fit(X, y)  # 선형 회귀 모델 훈련
lin_reg.predict(X_new)  # 새로운 입력 데이터에 대한 예측 수행
```

### 경사 하강법

- 여러 종류의 문제에서 최적의 해법을 찾는 일반적인 최적화 알고리즘
- 학습률 하이퍼파라미터로 결정되는 스텝의 크기가 중요
- 학습률이 너무 작으면 시간이 오래 걸림
- 학습률이 너무 크면 발산(Over-shooting)
- 경사 하강법의 종류
  - `배치 경사 하강법` : 매 스텝에서 **전체 훈련 세트**를 사용해 그레이디언트를 계산
  - `확률적 경사 하강법` : 매 스텝에서 한 개의 샘플을 **무작위**로 선택하고 그 하나의 샘플에 대한 그레이디언트를 계산
  - `미니배치 경사 하강법` : 미니배치라 부르는 임의의 **작은 샘플 세트**에 대해 그레이디언트를 계산

### Epoch란?

Epoch은 전체 데이터셋에 대해 한 번 순회하며 학습하는 과정이며,  
Epoch을 반복하여 모델을 학습시키고 최적화함.

Epoch이 너무 많으면 과적합이 발생하며,  
Epoch이 너무 적으면 데이터의 다양한 패턴을 학습하지 못함.

### 다항 회귀

비선형 데이터를 학습하기 위해 선형 모델을 사용하는 기법

### 학습 곡선

- `과소적합` : 훈련 세트와 교차 검증 점수 모두 낮은 경우
- `과대적합` : 훈련 세트에 대한 검증은 우수하지만 교차 검증 점수가 낮은 경우

### 릿지 회귀

규제가 추가된 선형 회귀 버전

### 라쏘 회귀

선형 회귀의 또 다른 규제된 버전  
가중치 값을 0으로 만듬

### 로지스틱 회귀

- `확률 모델`로서 독립변수의 선형 결합을 이용하여 사건의 발생 가능성을 예측하는 데 사용되는 통계 기법
- `이진 분류` 문제에 적용
- 샘플이 특정 클래스에 속할 확률을 추정하는 데 널리 사용
- `시그모이드 함수`를 사용하여 확률 추정

### 소프트맥스 회귀

- 로지스틱 회귀 모델을 일반화하여 `다중 클래스 분류`를 지원하도록 한 회귀 모델

### 미니배치 경사 하강법 실습

```python
theta_path_mgd = []  # 미니배치 경사 하강법을 통해 업데이트되는 theta 값들을 저장할 리스트
n_iterations = 50  # 전체 데이터에 대한 반복 횟수
minibatch_size = 20  # 미니배치 시 처리할 데이터의 크기
np.random.seed(42)  # 실행 결과를 동일하게 유지하기 위해 시드 설정
theta = np.random.randn(2, 1)  # 랜덤 초기화된 모델 파라미터 theta
t0, t1 = 200, 1000  # 학습 스케줄링 파라미터 설정

def learning_schedule(t):  # 학습 스케줄링 함수 정의
    return t0 / (t + t1)  # 반복 횟수 t에 따라 학습률을 반환

t = 0  # 반복 횟수 초기화
for epoch in range(n_iterations):  # 전체 데이터에 대해 반복
    shuffled_indices = np.random.permutation(m)  # 데이터를 무작위로 섞음
    X_b_shuffled = X_b[shuffled_indices]  # 섞인 순서대로 X_b(편향 추가된 입력 데이터) 정렬
    y_shuffled = y[shuffled_indices]  # 섞인 순서대로 y(목표 값) 정렬
    for i in range(0, m, minibatch_size):  # 미니배치 크기만큼 반복
        t += 1  # 반복 횟수 증가
        xi = X_b_shuffled[i:i+minibatch_size]  # 현재 미니배치의 입력 데이터
        yi = y_shuffled[i:i+minibatch_size]  # 현재 미니배치의 목표 값
        gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)  # 현재 미니배치에 대한 기울기 계산
        eta = learning_schedule(t)  # 학습률을 학습 스케줄링을 통해 설정
        theta = theta - eta * gradients  # theta 업데이트
        theta_path_mgd.append(theta)  # 업데이트된 theta 값 저장
```

