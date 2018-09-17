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
