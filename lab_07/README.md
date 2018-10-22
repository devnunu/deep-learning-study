# Learning rate, overfitting, 그리고 일반화(Regularization)

## Learning rate

- Gradient descent 에서 learning rate 이 너무 크면 값이 고정되지 않고 튕겨나가는 현상이 발생함(overshooting)
- Gradient descent 에서 learning rate 이 너무 작으면 목표점까지 도달하는 시간이 너무 오래 걸린다.

## Data(X) preprocessing for gradient descent

- 데이터 값에 큰 차이가 있을 경우에는 normalize 해야한다
- 만약 learning rate 를 잘 잡았는데 데이터 값이 발산하는 등의 모습을 보이면, 트레이닝되는 데이터 셋을 의심해보자

![2018-10-22 11 13 35](https://user-images.githubusercontent.com/20614643/47297510-3afa2580-d650-11e8-9cfe-a1ee8754174e.png)

- 계산 방법은, 평균과 분산의 비이다.

```py
x_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
```

## Overfitting

- 실제 트레이닝 셋과 너무 잘 맞는 모델을 생성하면, 해당 데이터셋에대한 결과값을 잘 나오나, 랜덤값을 넣었을때 결과가 다른경우가 있다.
- 트레이닝 데이터가 많을수록 오버 피팅을 줄일수 있다.
- 피처를 줄이거나 일반화(Regularization)시켜도 오버 피팅을 줄일수 있다.

## Regularization

![2018-10-22 11 21 48](https://user-images.githubusercontent.com/20614643/47297856-4732b280-d651-11e8-9b51-56fe25b518a2.png)

- 각각의 Element를 제곱하여 합한것에 상수(Regularization strength)를 곱해준다.
- 이값을 cost function에 더해준다.

```py
l2reg = 0.001 * tf.reduce_sum(tf.square(W))
```