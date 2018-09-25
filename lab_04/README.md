# multi-variable linear regression

## 변수가 여러개일때(multi-variable)

![image](https://user-images.githubusercontent.com/20614643/45617728-2dc8a480-baae-11e8-820a-db0bb55c39ee.png)

하나의 input 을 가지고 예측값을 구하는 예는 위와 같다.

![image](https://user-images.githubusercontent.com/20614643/45617745-3faa4780-baae-11e8-932e-fea2fdd90074.png)

그러나 실제 상황에서는 여러개의 input 을 가지는 경우가 많다. 위의 예제에서도 3 개의 변수를 가지고 있는데, 이럴때는 어떤식으로 예측값을 구해야할까?

![image](https://user-images.githubusercontent.com/20614643/45617806-74b69a00-baae-11e8-8b70-bde72d0e2b7d.png)

이렵지 않게, 여러개의 변수를 추가하는 것만으로 예측값을 구할 수 있다. 3 개가 아니라 더 많은 변수가 추가 되더라도 원리는 같다.

## 매트릭스(matrix)

![image](https://user-images.githubusercontent.com/20614643/45617858-a16ab180-baae-11e8-8544-114a7c80eb04.png)

그러나, 변수가 많아질수록 식이 길어지므로, 많이 불편해지게 된다. 이때, 매트릭스를 사용하면 식이 간단해 진다.

![image](https://user-images.githubusercontent.com/20614643/45617887-ba736280-baae-11e8-9239-a5a92bef72a4.png)

결론적으로 우리의 hypothesis 를 위와 같이 표현할 수 있다. 긴 식을 축약하여 표현한다는 것이 매트릭스의 가장 큰 장점이다. 매트릭스를 사용할때 보통 X 를 앞에 적고 W 를 뒤에 적는다. 또한 대문자로 표현하는 이유는 매트릭스라는 의미이다.

## 실제 예시

![image](https://user-images.githubusercontent.com/20614643/45617965-14742800-baaf-11e8-872e-29355eb61a95.png)

위와 같은 데이터 셋이 주어졌을 때 row 를 instance 라고 부른다. 위 그림은 5 개의 instance 를 가진다.

![image](https://user-images.githubusercontent.com/20614643/45618032-4eddc500-baaf-11e8-8616-dceff7936d7b.png)

이때, 인스턴스의 수 대로 만들어진 매트릭스와, w 를 그대로 곱하면 위와 같이, 연산값이 한번에 나오게된다. 즉, 전체 연산이 나오게 되는 것이다.

![image](https://user-images.githubusercontent.com/20614643/45618088-88aecb80-baaf-11e8-9fbe-14a4a1d68e11.png)

매트리스 곱에 의한 항 소거에 따라 [5,3]과 [3,1]의 결과값은 [5,1]이 된다.

![image](https://user-images.githubusercontent.com/20614643/45618178-daefec80-baaf-11e8-9edb-1b97d134a801.png)

보통 X 의 값은 주어진다. [5,3]에서 3 은 x variable 의 갯수이며, 5 는 instance 의 갯수이다. 출력값도 [5,1]에서 5 는 instance 이며, 1 은 결과값의 수이다(주로 1 로 고정). 그렇다면 w 는 어떻게 구할까?

W 는 X 의 variable 갯수와 H(X)의 결과값의 조합으로 이루어진다. 따라서 위 예제에서는 [3,1]이 된다.

### n 개의 output

![image](https://user-images.githubusercontent.com/20614643/45618316-3de18380-bab0-11e8-8c0b-f382f53f7cb4.png)

다음은 n 개의 output 에 대한 예시이다.

# Lab

## exampel 1

아래에 나오는 예제는 매트릭스를 사용하지 않고 구현된 multi-regression 코드이다.

### set_random_seed

```py
tf.set_random_seed(777)  # for reproducibility
```

random 넘버는 의사 난수이다. 따라서 seed 가 존재하는 경우 매번 동일한 값을 내 놓는다. 재현성을 위하여 seed 를 설정했다.
(https://www.tensorflow.org/api_docs/python/tf/set_random_seed)

### data set

![image](https://user-images.githubusercontent.com/20614643/45617965-14742800-baaf-11e8-872e-29355eb61a95.png)

위 그림을 파이썬으로 표현하면 다음과 같은 데이터 셋이 나온다

```py
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]

y_data = [152., 185., 180., 196., 142.]
```

### place holder

```py
# placeholders for a tensor that will be always fed.
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)
```

x 와 y 값은 실행되는 시점에 리스트로써 적용되며, 이를 위해 placeholder 로 선언해야한다.

```py
sess.run([cost, hypothesis, train], feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
```

다시 말해 session 을 run 하는 시점에서 feed_dict 넘겨준 데이터로 결정되는 것이다.

### w

```py
w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')
```

Variable 은 첫번째로 초기값을 받는데, 위에서는 1 개짜리 랜덤 리스트 형식을 변수로 생성한다. random_normal 의 첫번째 인자는 shape 인데 [1]이라는 뜻은 1 개짜리 리스트를 의미한다.

Variable 은 tf.global_variables_initializer()가 동반되어야하는데, 텐서플로의 그래프는 초기화 함수가 실행되어야 값이 할당되기 때문이다. 이 함수의 호출로 인해 변수가 초기화/할당된다.

### hypothesis

![image](https://user-images.githubusercontent.com/20614643/45617806-74b69a00-baae-11e8-8b70-bde72d0e2b7d.png)

```py
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b
```

hypothesis 는 위에서 본 그림과 같이 모든 w 와 x 를 곱하여 bias 를 더한다.

### optimizer

```
cost = tf.reduce_mean(tf.square(hypothesis - Y))
```

hypothesis 에 Y 를 뺸 값에 제곱을 한후 평균을 구한다. 이것이 cost function 이 된다

```py
# Minimize. Need a very small learning rate for this data set
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)
```

optimizer 는 GradientDescentOptimizer 로 선언하고, 이 때 최소화 된 값을 찾기 위해 cost function 을 인자로 넣어준다.

### 확인

```py
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
```

10 개의 스텝마다 cost, hypohesis, train 의 값을 확인 한다.

## example 2

데이터 셋을 2 차원으로 선언한다.

```py
x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]
```

### placeholder/ variable

```py
# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
```

placeholder 에서 element 는 고정이며, instance 는 무한이므로 [None, 고정값]의 shape 를 가진다.

W 의 shape 는 x 와 y 를 기반으로 계산되며 b 는 1 개의 리스트이다.

### matmul

```
hypothesis = tf.matmul(X, W) + b
```

matmul 은 행렬의 곱을 나타낸다(matrix multiply)
(https://www.tensorflow.org/api_docs/python/tf/matmul)

이후 과정은 동일하다.

## example 3

아래 코드는 numpy 를 이용해서 데이터를 호출하고 자르는 예제이다

```py
xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
```

csv 파일을 가져 온 후 numpy 의 slice 기능을 사용하여 x 와 y 의 데이터 셋으로 변경한다.

## example 4

```py
filename_queue = tf.train.string_input_producer(
    ['data-01-test-score.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# collect batches of csv in
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)
```

```py
# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

coord.request_stop()
coord.join(threads)
```
