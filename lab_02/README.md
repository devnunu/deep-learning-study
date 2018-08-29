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

## 실습

```py
import tensorflow as tf

# X and Y data
x_train = [1,2,3]
y_train = [1,2,3]

# 여기서 variable은 tensorflow가 사용하는 값이라고 볼 수 있다.
# 또는 trainable 한 값이라고 봐도 무방하다.
# variable을 만들때는 shape을 정의하고 값을 준다.
# 따라서 W,b의 값을 모르기때문에 랜덤값을 주게되는데, 이를위해 random_normal 함수를 사용한다.
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# W와 b 값이 정의 되었으므로 linear regression의 수식으로 hypothesis는 우리의 node가된다.
# Our hypothesis XW+b
hypothesis = x_train * W + b

# tf는 square라는 제곱 함수를 제공해준다.
# reduce_mean은 어떠한 텐서가 주어졌을때 평균을 내주는 함수이다
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# 텐서플로우에는 옵티마이저를 정의하는 여러가지 방법이 있는데 현재는 우선 GradientDescentOptimizer를 사용.
# 옵티마이저 정의 부분은 추후에 설명한다.
# minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
# 우리는 W,b라는 variable을 사용했기때문에 global_variables_initializer를 사용해줘야한다
# global_variables_initializer를 사용하면 변수를 initialize한다
# Initializes global varialbes in the graph
sess.run(tf.global_variables_initializer())

# Fit the Line
for step in range(2001):
    # node 실행
    sess.run(train)
    # 20번에 한번 꼴로 값들을 출력
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
```

위의 코드가 예제 코드이다. 현재 폴더 내의 jupyter 파일에서 결과를 확인 하도록 하자.

```
0 11.744587 [-0.37921482] [-0.4783018]
20 0.107978046 [0.82807183] [0.04675052]
40 0.0023352334 [0.9449368] [0.09240815]
60 0.0012529177 [0.957883] [0.09262203]
80 0.0011300528 [0.9608487] [0.08870305]
...
...
1900 1.7728729e-07 [0.99951094] [0.00111164]
1920 1.6100006e-07 [0.9995339] [0.00105942]
1940 1.4624108e-07 [0.9995557] [0.00100969]
1960 1.3284534e-07 [0.99957657] [0.00096229]
1980 1.2066369e-07 [0.99959654] [0.00091711]
2000 1.09611676e-07 [0.99961555] [0.00087405]
```

x_train = [1,2,3], y_train = [1,2,3] 이므로 1 차 방정식인 cost function 의 미지수 W 와 b 값은 사실 1 과 0 이다. 랜덤으로 발생한 값이라 초기에는 불안정 하나, 학습이 진행될수록 1 과 0 에 근접함을 볼 수 있다.

### Place holde

```py
import tensorflow as tf

# X and Y data
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis XW+b
hypothesis = X * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
# Initializes global varialbes in the graph
sess.run(tf.global_variables_initializer())

# Fit the Line
for step in range(2001):
    # sess.run에서 [](리스트)로 묶으면 한꺼번에 실행이 가능하다.
    # train에서 나오는 value는 중요하지 않으므로 _ 처리함
    cost_val, W_val, b_val, _ = sess.run([cost, W,b,train], feed_dict={X:[1,2,3],Y:[1,2,3]})
    # 20번에 한번 꼴로 값들을 출력
    if step % 20 == 0:
        print(step, cost_val,W_val, b_val)
```

위와 같은 동작을 하나, 이번에는 place holder 를 사용했다. 이전 강의에서 살펴보았듯이 place holder 를 사용하면 sess.run 이 일어나는 시점에 데이터를 삽입할 수 있다.
place holder 를 이용하는 가장 큰 이유중 하나는 만들어진 모델에 대해 값을 따로 넘겨줄 수 있다는 것이다. 즉 모델을 미리 구상하고, 학습때 값을 넘길수 있다. 또한 shape 도 전달이 가능하다.
[None]의 뜻한 [] -> 1 차원 배열, None -> 무한대의 갯수로 들어올 수 있다는 뜻이다.

placeholder 의 데이터를 {X:[1,2,3,4,5], Y:[2.1,3.1,4.1,5.1,6.1]}으로 변형시켜 실습해보도록하자.

### 예측값 구하기

```py
print(sess.run(hypothesis, feed_dict={X:[5]}))
print(sess.run(hypothesis, feed_dict={X:[2.5]}))
print(sess.run(hypothesis, feed_dict={X:[1.5,3.5]}))
```

모델에 데이터를 학습 시켰으므로, 다음값들을 넣어 출력으로 나오는 예측값이 실제 값과 어느정도 유사한지 비교해보자.