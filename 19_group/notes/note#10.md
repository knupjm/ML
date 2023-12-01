### 합성곱을 사용한 컴퓨터 비전

- `합성곱` : 이미지의 특징을 추출하는 데 사용되는 수학적 연산
- `합성곱 신경망` : 이미지를 분류하는 데 사용되는 신경망
- `컴퓨터 비전` : 컴퓨터가 이미지를 이해하는 것

### CNN(Convolutional Neural Network)

- `CNN` : 합성곱 신경망의 한 종류
- `합성곱층` : 합성곱 연산을 수행하는 층
- `풀링층` : 합성곱층의 출력을 입력으로 받아서 출력을 만드는 층
- `완전 연결층` : 풀링층의 출력을 입력으로 받아서 출력을 만드는 층

### 합성곱 연산

- `필터` : 합성곱 연산에 사용되는 행렬
- `특성맵` : 합성곱층의 출력
- `채널` : 특성맵의 깊이

```python
from sklearn.datasets import load_sample_image

china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255
images = np.array([china, flower])
batch_size, height, width, channels = images.shape

filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)  # 7*7 필터를 2개 만듬
filters[:, 3, :, 0] = 1  # 첫번째 필터는 수직선 감지
filters[3, :, :, 1] = 1  # 두번째 필터는 수평선 감지

# 2D 합성곱
outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")

for image_index in (0, 1):
    for feature_map_index in (0, 1):
        plt.subplot(2, 2, image_index * 2 + feature_map_index + 1)
        plt.axis("off")
        plt.imshow(outputs[image_index, :, :, feature_map_index], cmap="gray")
```

- `tf.nn.conv2d()` : 2D 합성곱을 수행하는 함수

### 풀링층

- `풀링` : 특성맵의 크기를 줄이는 작업
- `풀링층` : 풀링을 수행하는 층
- `최대 풀링` : 특성맵의 크기를 줄이기 위해 최대값을 계산하는 풀링
  - 파라미터 수를 획기적으로 줄여줌 : 계산량 감소
  - 작은 변화에 덜 민감해짐 : 과대적합 감소
- `평균 풀링` : 특성맵의 크기를 줄이기 위해 평균을 계산하는 풀링
  - 최대 풀링보다 성능이 떨어짐

### CNN 구조

- 네트워크를 통과하여 진행할수록 이미지는 점점 작아지지만,
  - 합성곱 층 때문에 일반적으로 점점 더 깊어짐 (즉, 더 많은 특성 맵을 가지게 됨)
- 이미지 인식 문제에서 완전 연결 층의 심층 신경망을 사용하지 않음.
  - 파라미터가 너무 많아지기 때문
- 합성곱 층에 사용하는 커널 크기
  - 작은 커널이 파라미터와 계산량이 적고 일반적으로 더 나은 성능을 냄.

```python
model = keras.models.Sequential(
    [
        DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),  # 28*28*1
        keras.layers.MaxPooling2D(pool_size=2),  # 14*14*64
        DefaultConv2D(filters=128),  # 14*14*128
        DefaultConv2D(filters=128),  # 14*14*128
        keras.layers.MaxPooling2D(pool_size=2),  # 7*7*128
        DefaultConv2D(filters=256),  # 7*7*256
        DefaultConv2D(filters=256),  # 7*7*256
        keras.layers.MaxPooling2D(pool_size=2),  # 4*4*256
        keras.layers.Flatten(),  # 4096
        keras.layers.Dense(units=128, activation="relu"),  # 128
        keras.layers.Dropout(0.5),  # 128
        keras.layers.Dense(units=64, activation="relu"),  # 64
        keras.layers.Dropout(0.5),  # 64
        keras.layers.Dense(units=10, activation="softmax"),  # 10
    ]
)
```

### 다양한 CNN 구조

- `LeNet-5` : 1998년에 소개된 최초의 CNN 구조
- `AlexNet` : 2012년에 소개된 CNN 구조
- `GoogLeNet` : 2014년에 소개된 CNN 구조
- `VGGNet` : 2014년에 소개된 CNN 구조
- `ResNet` : 2015년에 소개된 CNN 구조
- `Xception` : 2016년에 소개된 CNN 구조
- `SENet` : 2017년에 소개된 CNN 구조
- `MobileNet` : 2017년에 소개된 CNN 구조
