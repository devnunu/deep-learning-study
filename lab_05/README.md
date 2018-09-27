# Logistic classification

classification 중에서 상당히 정확도가 높은 알고리즘

## 복습

### Linear Regression

![image](https://user-images.githubusercontent.com/20614643/45993974-81e61100-c0cc-11e8-8efc-3c7253cf99c6.png)

## binary classification

- 둘 중 하나를 선택함
- 예) 이메일이 스팸이거나 아니거나, 좋아하는 타임라인 게시물을 보여주거나 아니거나, 신용카드의 사용내역이 가짜인지 아닌지.

## 0,1 encoding

- binary classification 를 기계적으로 표현하기 위해서는 예제와 다르게 0,1 으로 표현한다.
- 이를 통해 아래와 같은 문제를 해결 할 수 있다.

![image](https://user-images.githubusercontent.com/20614643/45994106-26685300-c0cd-11e8-96e0-fa3a3a3fb46a.png)

![image](https://user-images.githubusercontent.com/20614643/45994116-2cf6ca80-c0cd-11e8-97ec-c7052ecbc24b.png)

## linear regression 으로 할 수 없는 이유

- 0,1 로 결과값이 정해지기 때문에 입력과 결과가 비례하여 올라가지 않는다.
- 따라서 오차가 발생하며 정확한 모델이 생성되지 않는다.
- 또한 linear regression 으로 학습할 경우, 0,1 이외의 값이 나온다.

## Logistic Hypothesis

- 그러므로 결과값이 0 에서 1 사이로 나오는 함수를 찾아야한다.
- 결과값이 0 에서 1 사이인 s 자 곡선을 sigmoid function 또는 logistic function 이라고 한다.
- 식은 아래와 같다

![image](https://user-images.githubusercontent.com/20614643/45994383-5106db80-c0ce-11e8-9089-eeaa7aa4bc8f.png)

## 기존의 cost function 에 Logistic Hypothesis 대입시 문제점

- 기존 linear 함수와 다르게 Logistic Hypothesis 의 cost function 은 구불구불한 곡선으로 이루어진다.
- 이 때문에 시작한 점에 따라 끝점이 달리지고, 이는 local minimum point 가 된다.
- 결과적으로 이러한 cost function 을 사용할 수는 없다.

![image](https://user-images.githubusercontent.com/20614643/45994559-29fcd980-c0cf-11e8-9ae3-175719316dca.png)

## 새로운 cost function

- 2 가지 케이스로 나누어 계산한다.
- 예측이 실패 했을때 코스트 값이 무한대로 발산하기 때문에, 계산이 쉬워진다.

![image](https://user-images.githubusercontent.com/20614643/45996328-d0001200-c0d6-11e8-99ba-2e40dcfda2d1.png)

- 위의 식을 그대로 텐서 플로로 옮기면 if 문이 들어가야한다.
- 따라서 아래와 같이 간단하게 한줄의 수식으로 합칠수 있다.

![image](https://user-images.githubusercontent.com/20614643/45996613-c1fec100-c0d7-11e8-8fec-9c0ae2a68e7e.png)

- Gradient decent aligorithm 의 경우 미분하여 값을 구현 할 수 있다.

## 실습

우리는 다음과 같은 함수를 소스코드로 표현할 것이다.

![image](https://user-images.githubusercontent.com/20614643/46004741-8fac8e00-c0ee-11e8-9084-e4ef40b54037.png)

### 데이터 및 변수 선언

```py
x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
```

- x 의 데이터는 2 차원 배열으로, y 는 pass/fail 으로 선언한다.
- 또한 X 와 Y 가 들어갈 place holder 를 선언하고, shape 를 넣어준다.
- 매트릭스 곱이므로 w 와 b 의 모양을 이와 맞게 설정해준다.

## hypothesis

```py
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
```

- sigmoid 는 tf.div(1., 1. + tf.exp(tf.matmul(X, W) + b))와 같다.

## cost

```py
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
```

- 만들어진 수식과 동일하게 처리한다.

## predict

```py
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
```

- 보통 0.5 보다 크면 pass, 작으면 fail이다.
- dtype으로 캐스팅하면 pass/fail 값이 나온다.
- accuracy은 예측값의 평균값이라고 할 수 있다. 즉, n번중에 몇번 맞추었는지의 비율이다.

## 실제 데이터에 적용(당뇨병 예측)

어떤 형태의 값들이 있고, 마지막에 당뇨병인지 아닌지 결과 값이 들어있다.

이 데이터를 적용했을때 당뇨병인지 아닌지에 대해 example2에서 알아보자