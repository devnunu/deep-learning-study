# Linear Regression 의 Hypothesis 와 cost function

## Linear Regression(선형 회귀)

<img src="https://user-images.githubusercontent.com/20614643/44731554-619f5280-ab1e-11e8-912d-72d35042eeb4.png" />

- 위의 그림 처럼 공부한 시간 대비 성적에 관한 데이터를 바탕으로 supervised learn 을 진행 한다.
- 성적의 결과값은 0~100 의 넓은 분포도를 보이므로 supervised learn 의 유형중, regression 에 속한다.
- 이 데이터를 학습시키는것을 training, 학습시키는 데이터를 training data 라고 한다.
- 학습된 결과는 regression 이라는 모델로 형성이 된다.
- 유저는 시간(hour)이라는 x 값을 주고 성적이라는 **예측값 y**를 받을 수 있다.
- 따라서 이를 Linear Regression 이라고 한다.

## Hypothesis(가설)

<img src="https://user-images.githubusercontent.com/20614643/44731838-0457d100-ab1f-11e8-8e84-81eececbd0cf.png" />

위의 데이터 셋을 기본으로 Liear Regression 을 진행해보자

<img src="https://user-images.githubusercontent.com/20614643/44731887-1fc2dc00-ab1f-11e8-9bfd-618b2fa7551d.png" />

주어진 데이터 셋을 그래프로 표현한 결과이다. 그래프는 x 값에 대응하는 y 값이 나오는데, 선형 그래프의 형태를 따르므로 Liear 하다.
세상에 있는 많은 데이터, 현상들이 Linear 하게 나타는 경우는 많이 존재한다. 예를 들어, 숙면시간 대비 달리기 또는 집의 크기와 가격의 경우가 있다. 따라서 Linear 한 가설을 세운다는 것은 위의 그래프와 마찬가지로 데이터 셋에 맞는 선형의 그래프를 찾는 것이다.

```
H(x) = Wx + b
```

즉 수식으로는 위와 같이 표현된다. H(x)는 우리의 가설(Hypothesis)이며, W 와 x, b 의 값에 따라 여러 형태의 선이 나타난다.

## cost function과 최소 제곱법

<img src="https://user-images.githubusercontent.com/20614643/44733649-fad06800-ab22-11e8-8e03-7e82f49bb2a3.png" />

이렇게 나타난 선들 중에서 어떤 선이 우리가 가지고 있는 데이터와 잘 맞는것이지 알아내야한다. 즉, 선의 형태를 위한 W 나 b 를 찾는것이다.

```
H(x) = 1 * x + 0
H(x) = 0.5 * x + 2
H(x) = 3 * x + 9
```

우리는 Linear Regression function(선형 회귀 함수)의 적절한 식을 찾기 위하여, 위와 같은 Hypothesis(가설)을 수식으로 세울수 있다. 그럼 이렇게 세운 수식 중에 무엇이 가장 정확한 식으로 선택되는 것일까.

<img src="https://user-images.githubusercontent.com/20614643/44732917-5994e200-ab21-11e8-956d-4de749a2d3ed.png" />

어떤 Hypothesis 가 좋은가를 판별할때는 선형 그래프와 실제 데이터 셋과의 거리를 계산한다. 물론 점과 선과의 거리가 가까울수록 더 좋은 것이다. 우리가 계산한것이 실제 데이터와 얼마나 다른가 거리를 계산하는 것을 Cost function 또는 Loss function 이라고 부른다

```
// bad
H(x) - y

// good
(H(x) - y)^2
```

거리 계산은 위와 같이 수식으로 표현된다. 차이는 + 또는 -가 될 수 있기 때문에 제곱을 하면 차이를 양수로 표현 할 수 있다. 또한 차이가 클때 제곱되면 차이가 더 커지므로 패널티를 부여할 수 있다. 이를 **최소 제곱법**이라고 한다.

<img src="https://user-images.githubusercontent.com/20614643/44733174-f6f01600-ab21-11e8-8ac6-a62d952b2e04.png" />

즉, 모든 값들의 차이(거리)를 더해 평균을 구하는 것이 cost function 이라고 할 수 있다.

<img src="https://user-images.githubusercontent.com/20614643/44733357-6403ab80-ab22-11e8-8363-8118a52fe53a.png" />

정리하자면 위와 같은 수식이 나오며, 앞서 언급했듯이 cost function 의 가장 작은 값(Minimize cost)을 출력하는 W 와 b 를 구하는 것이 목적이다.
