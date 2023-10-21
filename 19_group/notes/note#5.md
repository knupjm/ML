# 기계학습 - 7주차

### 결정 트리

- **분류와 회귀** 문제에 널리 사용하는 모델
- **다중 출력**을 지원하는 결정 트리도 존재
- **랜덤 포레스트**의 기본 구성 요소로 사용
- 복잡한 데이터셋도 학습할 수 있어서 **전처리 과정**이 필요 없음
- Depth가 깊으면 확률값이 떨어짐
  - 결정 경계가 복잡해짐
  - 노이즈에 민감해짐
  - 과대적합이 될 확률이 높음
  - 때문에 규제 매개변수가 필요함

### CART 훈련 알고리즘

- `CART`: Classification And Regression Tree(분류와 회귀 트리)
- `탐욕적 알고리즘` : 각 단계에서 최적의 솔루션을 찾음

### 지니 불순도 또는 엔트로피 불순도

- 기본적으로 지니 불순도를 사용
- 엔트로피 불순도를 사용하고 싶다면 `criterion` 매개변수를 `entropy`로 지정
- 둘 다 큰 차이가 없어서 비슷한 트리를 만들어줌
- 지니 불순도가 좀 더 계산이 빠름
- 지니 불순도는 한쪽 가지를 고립시키는 경향이 있음
- 엔트로피 불순도는 조금 더 균형 잡힌 트리를 만듬

### 규제 매개변수

- `min_samples_leaf` : 리프 노드가 가지고 있어야 할 최소 샘플 수
- `min_samples_split` : 분할되기 위해 노드가 가지고 있어야 할 최소 샘플 수
- `min_weight_fraction_leaf` : 가중치가 부여된 전체 샘플 수에서의 비율
- `max_leaf_nodes` : 리프 노드의 최대 수
- `max_features` : 각 노드에서 분할에 사용할 특성의 최대 수

### 연습 문제

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- `make_moons` : 초승달 모양의 데이터셋을 만듬
- `train_test_split` : 데이터를 훈련 세트와 테스트 세트로 나눔

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, KFold

classifier = DecisionTreeClassifier()
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(classifier, param_grid, cv=kfold, scoring='accuracy')
grid_search.fit(X, y)
```

- `DecisionTreeClassifier` : 결정 트리 분류기
- `param_grid` : 탐색할 매개변수의 종류
- `KFlod` : 교차 검증을 위한 폴드 생성기
- `GridSearchCV` : 교차 검증과 그리드 탐색을 동시에 수행

```python
grid_search.best_estimator_
```

<pre>
DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=4)
</pre>

최적 모델 출력

```python
grid_search.best_params_
```

<pre>
{'criterion': 'entropy',
 'max_depth': 5,
 'min_samples_leaf': 4,
 'min_samples_split': 2}
</pre>

최적 모델의 최적 매개변수 출력

```python
grid_search.best_score_
```

<pre>
0.8550000000000001
</pre>

최적 모델의 평가 점수 출력

```python
from sklearn.metrics import accuracy_score

y_pred = grid_search.predict(X_test)
accuracy_score(y_test, y_pred)
```

<pre>
0.87
</pre>

`accuracy_score`로 테스트 세트를 사용해 최종 모델 평가
