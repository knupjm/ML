### 인공신경망

- `인공신경망` : 뇌에 있는 **뉴런**의 동작을 모방한 것
- `인공뉴런` : 하나 이상의 이진 입력과 하나의 이진 출력을 가짐

### 퍼셉트론

#### 퍼셉트론이란?

`퍼셉트론` : 간단한 인공신경망의 한 종류

#### TLU?

`TLU` : 퍼셉트론의 각 입력 연결에 가중치를 부여하는 것  
퍼셉트론은 하나의 층에 여러 개의 TLU가 모여있음

### 다층 퍼셉트론

`다층 퍼셉트론` : 퍼셉트론을 여러 개 쌓아올린 것  
`입력층` 하나와 `은닉층`이라는 하나 이상의 TLU층과 `출력층`으로 구성

- `완전연결층` : 모든 뉴런이 이전 층의 모든 뉴런과 연결되어 있는 층
- `심층신경망(DNN)` : 은닉층이 2개 이상인 신경망
- `역전파 훈련` : 신경망의 모든 연결 가중치에 대한 오차 함수의 그래디언트를 계산하고 이 그래디언트를 경사 하강법 단계에 사용하여 가중치를 수정하는 것

### 케라스 사용법

- `keras` : 신경망을 쉽게 구현할 수 있도록 도와주는 파이썬 라이브러리
- `keras.datasets` : 케라스에서 데이터셋을 불러오는 모듈
- `keras.models` : 케라스에서 모델을 만드는 모듈
- `keras.layers` : 케라스에서 층을 만드는 모듈

#### 케라스를 사용한 예제(이미지 분류)

```python
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data() # 훈련 세트와 테스트 세트로 나눔

X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255. # 0 ~ 1 사이로 정규화
```

```python
model = keras.models.Sequential(
    [keras.layers.Flatten(input_shape=[28, 28]), # 입력층
     keras.layers.Dense(300, activation="relu"), # 은닉층
     keras.layers.Dense(100, activation="relu"), # 은닉층
     keras.layers.Dense(10, activation="softmax")]) # 출력층
```

```python
model.compile(loss="sparse_categorical_crossentropy", # 손실 함수
              optimizer="sgd", # 옵티마이저
              metrics=["accuracy"]) # 성능 측정 지표
model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid)) # 훈련
model.evaluate(X_test, y_test) # 테스트 세트로 모델 평가

X_new = X_test[:3] # 새로운 샘플
y_proba = model.predict(X_new) # 샘플에 대한 예측 확률
y_proba.round(2) # 소수점 둘째 자리까지 반올림해서 확인
```

#### 케라스를 사용한 회귀 예제(캘리포니아 주택 가격)

```python
model = keras.models.Sequential(
    [keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]), # 입력층
     keras.layers.Dense(1)]) # 출력층
model.compile(loss='mean_squared_error', # 손실 함수
    optimizer=keras.optimizers.SGD(lr=1e-3)) # 옵티마이저
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid)) # 훈련
mse_test = model.evaluate(X_test, y_test) # 테스트 세트로 모델 평가
X_new = X_test[:3] # 새로운 샘플
y_pred = model.predict(X_new) # 샘플에 대한 예측
```

### 함수형 API

`함수형 API` : 케라스의 함수형 API를 사용하면 여러 개의 입력이나 출력을 가진 모델을 만들 수 있음

```python
input_A = keras.layers.Input(shape=[5], name="wide_input") # 입력층
input_B = keras.layers.Input(shape=[6], name="deep_input") # 입력층
hidden1 = keras.layers.Dense(30, activation="relu")(input_B) # 은닉층
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1) # 은닉층
concat = keras.layers.concatenate([input_A, hidden2]) # 입력층과 은닉층을 연결
output = keras.layers.Dense(1, name="output")(concat) # 출력층
model = keras.Model(inputs=[input_A, input_B], outputs=[output]) # 모델 생성
```

```python
model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3)) # 손실 함수와 옵티마이저 지정

X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:] # 훈련 세트를 두 개로 나눔
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:] # 검증 세트를 두 개로 나눔
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:] # 테스트 세트를 두 개로 나눔
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3] # 새로운 샘플

history = model.fit((X_train_A, X_train_B), y_train, epochs=20, validation_data=((X_valid_A, X_valid_B), y_valid)) # 훈련
mse_test = model.evaluate((X_test_A, X_test_B), y_test) # 테스트 세트로 모델 평가
y_pred = model.predict((X_new_A, X_new_B)) # 샘플에 대한 예측
```

### 모델 저장과 복원

```python
model.save("my_keras_model.h5") # 모델 저장
model = keras.models.load_model("my_keras_model.h5") # 모델 복원
```

> 서브클래스 API를 사용한 모델은 저장할 수 없음

### 콜백 사용하기

- `콜백` : 훈련 과정 중간에 어떤 작업을 수행할 수 있도록 도와주는 객체
- `ModelCheckpoint` : 훈련하는 동안 일정 간격으로 모델의 체크포인트를 저장
- `EarlyStopping` : 검증 세트에 대한 점수가 더 이상 향상되지 않으면 훈련을 중지
- `keras.callbacks.Callback`을 상속받아 직접 콜백을 만들 수도 있음

```python
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True) # 조기 종료 콜백
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[checkpoint_cb, early_stopping_cb]) # 훈련
mse_test = model.evaluate(X_test, y_test) # 테스트 세트로 모델 평가
```

### 텐서보드를 사용해 시각화하기

```python
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir) # 텐서보드 콜백
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid),
                    callbacks=[tensorboard_cb]) # 훈련
```

```bash
$ tensorboard --logdir=./my_logs --port=6006
```

[http://localhost:6006](http://localhost:6006) 확인

### 신경망 하이퍼파라미터 튜닝하기

- `하이퍼파라미터` : 은닉층의 수, 은닉층의 뉴런 수, 각 은닉층의 활성화 함수 등과 같은 학습 과정에 영향을 주는 파라미터
- `KerasRegressor` : 사이킷런의 `RandomizedSearchCV`나 `GridSearchCV`와 같은 사이킷런의 `RandomizedSearchCV`나 `GridSearchCV`와 같은 사이킷런의 교차 검증 기능을 사용하기 위해 사이킷런의 추정기와 동일하게 작동하는 케라스 모델을 감싸는 래퍼
- `RandomizedSearchCV` : 랜덤 탐색을 사용하여 하이퍼파라미터 탐색

### 신경망 하이퍼파리미터 튜닝하기 - 조사

- `베이지안 최적화` : 탐색 공간을 확률 모델로 모델링하고 이 모델을 사용하여 탐색 공간에서 가장 가능성이 높은 지점을 탐색하는 방법
  - `Hyperopt`, `kopt`, `skpot` 등의 라이브러리를 사용하여 구현 가능
- `유전 알고리즘` : 생명체의 진화 과정에서 사용되는 유전 알고리즘을 사용하여 하이퍼파라미터 탐색하는 방법
  - `DEAP`, `TPOT` 등의 라이브러리를 사용하여 구현 가능
