# 기계학습 - 과제 #3

### 1. 기계 학습에서 학습이란 무엇인지를 정리하시오

기계 학습에서 학습이란 주어진 데이터를 분석하여  
명시적인 프로그래밍 없이 컴퓨터가 스스로 학습하는 것을 뜻함.

### 2 확률적 경사 하강법의 소스 코드를 분석하시오

```python
n_epochs = 50
t0, t1 = 5, 50

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1) # 무작위 초기화

for epoch in range(n_epochs):
  for i in range(m):
    random_index = np.random.randint(m)
    xi = X-b[random_index:random_index+1]
    yi = y[random_index:random_index+1]
    gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
    eta = learning_schedule(epoch * m + i)
    theta = theta - eta * gradients
```

- `n_epochs` : 에포크 횟수
  - 에포크란 훈련 세트의 전체를 한 번 학습하는 단위
- `t0`, `t1` : 학습 스케줄 하이퍼파라미터
- `learning_schedule()` : 학습 스케줄 함수
- `random_index` : 무작위 인덱스
- `xi` : 무작위 샘플
- `yi` : 무작위 타깃
- `gradients` : 비용 함수의 그레이디언트 벡터
- `eta` : 학습률
- `theta` : 모델 파라미터

위 코드는 학습 스케줄을 이용한 확률적 경사 하강법의 소스 코드임.
