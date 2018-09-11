# Linear Regression 의 cost 최소화 알고리즘의 원리 설명

## cost(W)를 세부적으로 살펴보자

![image](https://user-images.githubusercontent.com/20614643/45330720-c092ac00-b5a1-11e8-846d-98b87b3db0c8.png)

![image](https://user-images.githubusercontent.com/20614643/45330778-149d9080-b5a2-11e8-9fa9-27efc742f13e.png)

cost function 에 값을 대입하여 값을 구하여, 그래프 상에 x 와 y 를 나타내면 아래와 같은 그림이 된다.

![image](https://user-images.githubusercontent.com/20614643/45330852-5f1f0d00-b5a2-11e8-92d0-cfa44413b4e8.png)

함수에서 최소의 값을 가지는 점을 구하는 것이 minimal cost function 의 값이다.

## Gradient descent algorithm

이 최소값을 찾기 위한 알고리즘이 Gradient descent algorithm이며, 직역하자면 경사를 따라 내려가는 알고리즘 이라고 할 수 있다.

- cost function 을 최소화 하기위해 사용됨
- cost function 뿐만 아니라 여러 문제를 최소화 하기 위해 사용됨

## Gradient descent algorithm의 예

가장 작은 포인트를 어떻게 찾을까?? 만약 한라산 정상에 올랐을떄 지상으로 가기 위해서는, 한 발짝씩 내려가는 방법 밖에 없다. 

Gradient descent algorithm도 이와 동일하게, 경사도를 따라서 한 포인트씩 이동한다. 따라서 최소값에 도달한다.

## 동작 원리

- 아무 지점에서 시작할 수 있다.
- W를 조금 변경하여 cost를 줄인다.
- 각 시간대의 경사도를 계산(미분)하고, 이를 반복한다.
- 어떤 지점에서 시작하건간에, 항상 최저점에 도달할 수 있다.

## Gradient descent algorithm 도출(경사도 계산)

![image](https://user-images.githubusercontent.com/20614643/45331024-5f6bd800-b5a3-11e8-8fa1-dde1f4856f1a.png)

수식을 간단하게 위해서 분모에 2를 곱해준다. 2를 곱해줘도 결과값에는 별 차이가 없으므로, 이는 계산의 편의를 위함이다.

![image](https://user-images.githubusercontent.com/20614643/45331042-7f030080-b5a3-11e8-9732-3fd7312225c6.png)

공식적인 알고리즘은 위와 같다. 알파는 기울기를 뜻하며, 기울기가 마이너스이면 앞의 마이너스를 만나 포인트가 양의 방향으로 움지이고, 기울기가 플러스이면 그 반대이다.

![image](https://user-images.githubusercontent.com/20614643/45331302-e0779f00-b5a4-11e8-8fd3-992b0056be7b.png)

위의 공식은 W로 미분의 과정을 보여준다. 가장 마지막 줄이 최종으로 얻을수 있는 값이다. 따라서 Gradient descent algorithm이란 가장 마지막 줄의 수식으로 표현이 가능하며

이를 cost function에 기계적으로 대입하면 최솟값을 구할수 있게 된다.

## Convex function

![image](https://user-images.githubusercontent.com/20614643/45331405-74496b00-b5a5-11e8-9dd7-98f1072952fc.png)

cost, w, b를 3축으로 3차원 그래프를 그렸을 때 위와 같은 그래프가 나오게된다. 이는 어느 점에서 출발 하던지 똑같은 결과 값을 얻게 된다는 것이다.

따라서 linear regression을 적용하기 전에 Convex funtion을 만족하는지 검증해야한다.

![image](https://user-images.githubusercontent.com/20614643/45331446-ae1a7180-b5a5-11e8-9e0b-a196fa93afcf.png)

만약 Convex function 이 아니라면 위의 그림과 같이 출발선과 경사의 차이에 따라 최솟값에 해당하는 결과가 달라진다.

