# softmax classification

## Multinomial classification

- 여러 개의 집단을 구분하기 위한 구분법.
- A, B, C 가 있을 때 A or Not, B or Not, C or Not 의 선을 구한다.
- 독립적으로 값을 구하면 복잡하므로 행렬로 한번에 구한다.

![2018-10-09 1 48 57](https://user-images.githubusercontent.com/20614643/46622433-8fd47100-cb65-11e8-8be5-94235e87150b.png)
![2018-10-09 1 49 57](https://user-images.githubusercontent.com/20614643/46622457-a8dd2200-cb65-11e8-8680-8a77ddfc3ec4.png)

## softmax

![2018-10-09 7 02 31](https://user-images.githubusercontent.com/20614643/46662263-e8534f00-cbf5-11e8-857d-31d85874a31b.png)

## cost function

cross entropy 함수를 통해 실제값과 예측값의 차이를 구한다.

![2018-10-09 7 02 31](https://user-images.githubusercontent.com/20614643/46662338-205a9200-cbf6-11e8-824d-9e5c65495f4a.png)

## 예제

텐서플로로 구현하는 것은 간단하다

![2018-10-09 7 02 31](https://user-images.githubusercontent.com/20614643/46671863-e8ad1380-cc10-11e8-83cd-80568367691e.png)

![2018-10-09 10 18 15](https://user-images.githubusercontent.com/20614643/46671983-3de92500-cc11-11e8-9373-c6c919c9a23b.png)

### 데이터 표시법

one hot encoding 을 사용한다. 즉 n 개의 레이블에 대해 해당하는 한가지만 1 으로 처리해주는 것이다.

### 출력

```py
    # Testing & One-hot encoding
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print(a, sess.run(tf.argmax(a, 1)))

    # [[1.3890496e-03 9.9860185e-01 9.0612912e-06]] [1]
```

자동으로 최대한 확률값을 구하는 tf.argmax 를 사용한다. tf.argmax 로 결과값을 구할 수 있다.

## fancy

### softmax_cross_entropy_with_logits

```py
# Cross entropy cost/loss 이전 버전
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

# Cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                 labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
```

위의 softmax_cross_entropy_with_logits 를 사용하면 좀 더 간결하게 표현이 가능하다.

### 데이터 값

```py
Y = tf.placeholder(tf.int32, [None,1])
Y_one_hot = tf.one_hot(Y,nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
```

- Y 는 int 형의 2 차원 배열로 나타난다.
- tf.one_hot 은 해당 배열을 one hot 의 형태로 변환해준다. 다만 이와 같은 형태로만 선언하면 아래에서 호출시 에러가 발생한다
  왜냐하면 one_hot 함수는 1 차원을 더해서 shape 가 추가되기 떄문이다.([?, 1] => [?, 1, 7])
- tf.reshape 로 기존에 만들어둔 one hot의 차원을 재정렬 해준다.

### 예측값

```py
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

- prediction은 결과적인 예측값이다.
- correct_prediction은 에측값이 맞았는지의 여부이다
- accuracy는 정확도이다.