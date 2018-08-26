<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [머신 러닝의 기본적인 용어와 개념 설명](#%EB%A8%B8%EC%8B%A0-%EB%9F%AC%EB%8B%9D%EC%9D%98-%EA%B8%B0%EB%B3%B8%EC%A0%81%EC%9D%B8-%EC%9A%A9%EC%96%B4%EC%99%80-%EA%B0%9C%EB%85%90-%EC%84%A4%EB%AA%85)
  - [머신 러닝?](#%EB%A8%B8%EC%8B%A0-%EB%9F%AC%EB%8B%9D)
    - [머신 러닝의 개념](#%EB%A8%B8%EC%8B%A0-%EB%9F%AC%EB%8B%9D%EC%9D%98-%EA%B0%9C%EB%85%90)
    - [Supervised/Unsupervised learning](#supervisedunsupervised-learning)
    - [Supervised learning](#supervised-learning)
    - [Supervised learning 의 종류](#supervised-learning-%EC%9D%98-%EC%A2%85%EB%A5%98)
    - [Traning data set](#traning-data-set)
    - [알파고?](#%EC%95%8C%ED%8C%8C%EA%B3%A0)
  - [TensorFlow 의 설치 및 기본적인 operations](#tensorflow-%EC%9D%98-%EC%84%A4%EC%B9%98-%EB%B0%8F-%EA%B8%B0%EB%B3%B8%EC%A0%81%EC%9D%B8-operations)
    - [텐서플로란 무엇인가?](#%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80)
    - [데이터 플로우 그래프란?](#%EB%8D%B0%EC%9D%B4%ED%84%B0-%ED%94%8C%EB%A1%9C%EC%9A%B0-%EA%B7%B8%EB%9E%98%ED%94%84%EB%9E%80)
    - [가상환경 설치 및 설정](#%EA%B0%80%EC%83%81%ED%99%98%EA%B2%BD-%EC%84%A4%EC%B9%98-%EB%B0%8F-%EC%84%A4%EC%A0%95)
    - [특정 버전 파이썬이 설치되지 않을 경우](#%ED%8A%B9%EC%A0%95-%EB%B2%84%EC%A0%84-%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9D%B4-%EC%84%A4%EC%B9%98%EB%90%98%EC%A7%80-%EC%95%8A%EC%9D%84-%EA%B2%BD%EC%9A%B0)
    - [특정 폴더 진입시 가상환경 자동실행 설정](#%ED%8A%B9%EC%A0%95-%ED%8F%B4%EB%8D%94-%EC%A7%84%EC%9E%85%EC%8B%9C-%EA%B0%80%EC%83%81%ED%99%98%EA%B2%BD-%EC%9E%90%EB%8F%99%EC%8B%A4%ED%96%89-%EC%84%A4%EC%A0%95)
    - [가상 환경(virtualenv, pyenv) 명령어](#%EA%B0%80%EC%83%81-%ED%99%98%EA%B2%BDvirtualenv-pyenv-%EB%AA%85%EB%A0%B9%EC%96%B4)
    - [텐서플로 설치 및 확인](#%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C-%EC%84%A4%EC%B9%98-%EB%B0%8F-%ED%99%95%EC%9D%B8)
  - [예제 소스코드](#%EC%98%88%EC%A0%9C-%EC%86%8C%EC%8A%A4%EC%BD%94%EB%93%9C)
  - [실습](#%EC%8B%A4%EC%8A%B5)
    - [주피터 노트북(jupyter notebook) 설치 및 실행](#%EC%A3%BC%ED%94%BC%ED%84%B0-%EB%85%B8%ED%8A%B8%EB%B6%81jupyter-notebook-%EC%84%A4%EC%B9%98-%EB%B0%8F-%EC%8B%A4%ED%96%89)
    - [TensorFlow Hello World](#tensorflow-hello-world)
    - [Computational Graph 1](#computational-graph-1)
    - [텐서플로 머신러닝](#%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D)
    - [Placeholder](#placeholder)
    - [Tensor Ranks, Shapes, and Types](#tensor-ranks-shapes-and-types)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# 머신 러닝의 기본적인 용어와 개념 설명

## 머신 러닝?

### 머신 러닝의 개념

주어진 환경과 조건 내에서 개발이 진행될 때 explicit programming 이라고 한다. 그러나 모든 프로그래밍 작업이 explicit 하지는 않는데, 예를 들어, 스팸 필터나 무인 주행 자동차는 예측할 수 없는 많은 규칙이 있다.

그래서 1959 년에 Arthur Samuel 이라는 사람이 다음과 같은 생각을 했다. 우리가 모든 조건을 일일히 프로그래밍 하는 것이 아니라, 주어진 자료나 현상에서 자동적으로 배우게 하는 것이다. 이것이 바로 머신러닝의 시초이다. 즉, 프로그래밍에 개발자가 관여하는 것이 아니라 주어진 자료를 바탕으로 학습하는것이 머신 러닝이라고 할 수 있다.

### Supervised/Unsupervised learning

머신러닝은 학습하는 방법에 따라 Supervised/Unsupervised learning 으로 나뉜다.

Supervised : label 들이 정해져있는 데이터(traning set)로 학습하는 것을 Supervised leaning 이라고 한다. 예를 들어, 이미지가 고양이/개 인지 판별하는 프로그램도 머신 러닝을 바탕으로 개발되는데, cat 또는 dog 라는 label 이 달려있는 이미지로 먼저 학습을 진행한 후 판별을 하게 된다.

Unsupervised : 일일히 label 을 달수 없는 자료가 있다. 예를 들면, Google news(비슷한 카테고리의 뉴스 수집)나 Word clustering(단어 수집)등이 있다. 이는 주어진 자료가 아닌 프로그램이 스스로 학습해야한다.

이중에서 Supervised learning 을 주로 다룰 예정이다.

### Supervised learning

Supervised learning 에서 다루는 주요한 문제점은 다음과 같다

- Image labeling : 이미지로부터 학습
- Email spam filter : 스팸인지 아닌지 구분된 이메일로 부터 학습
- Predicting exam score : 이전에 시험을 친 사람들의 준비시간 대비 성적으로 자신의 기대 성적을 예측

### Supervised learning 의 종류

Supervised learning 를 다루는 경우 케이스는 대개 3 가지로 압축된다.

- 시험 공부 시간 대비 최종 시험의 점수 : 0 ~ 100 점 까지 넓은 분포도를 보이는 경우 regression 이라고 한다.
- 시험 공부 시간 대비 최종 시험의 결과(Pass/non-Pass) : 결과를 2 가지로 예측하는 경우 binary classification 라고 한다.
- 시험 공부 시간 대비 최종 시험의 등급(A,B,C,E and F) : 결과가 다중인 경우 multi-label classification 라고 한다.

<p align="center">
<img src="https://user-images.githubusercontent.com/20614643/44619947-f1978f00-a8c7-11e8-8b91-89a709b6faa5.png" width="150px"/>
<img src="https://user-images.githubusercontent.com/20614643/44619948-f1978f00-a8c7-11e8-8be7-2397f346ebb5.png" width="150px"/>
<img src="https://user-images.githubusercontent.com/20614643/44619934-8e0d6180-a8c7-11e8-83ee-84875b449b05.png" width="150px"/>
<p/>

### Traning data set

Supervised learning 은 다음과 같은 값들을 주로 사용한다.

- ML(머신 러닝): 모델
- Y(label): 정해진 값
- X: 특징(feature)

데이터 셋은 Y 와 X 로 구성되므로, 특징에 따라 정해진 값을 학습하여 모델이 도출 된다.
이 때, 임의의 X 값을 구성된 모델에 대입 했을 때, Y 의 예상값을 구하는게 일반적인 Supervised learning 의 예이다.
때문에 모델의 구성을 위해 미리 준비된 Traning data set 이 필요하다.

### 알파고?

앞서 든 예와 같이 알파고는 다음과 같은 순서로 동작했다.

1. 기존의 사람들이 바둑을 둔 기보를 학습한다.
2. 이세돌이 돌을 착수 한다
3. 이 데이터를 입력 받아 모델에 적용한 후 결과값을 내놓는다.

## TensorFlow 의 설치 및 기본적인 operations

### 텐서플로란 무엇인가?

텐서플로란 구글에서 만든 오픈소스 라이브러리이며, 기계 학습을 위해 개발 되었다.

한마디로 정의하자면, 파이썬을 기반으로 데이터 플로우 그래프를 사용하여 수적인 계산을 할 수 있는 오픈소스 라이브러리이다.

텐서플로 말고도 사용할 수 있는 라이브러리는 많다. 그러나 모든 라이브러리를 비교해 보았을 때, 컨트리뷰트나, 참여자, 레퍼런스 등에서 압도적으로 높은 점수를 가지고 있기때문에, 초반 학습에 많은 이점이 있다.

### 데이터 플로우 그래프란?

그래프는 노드와 노드를 연결하는 엣지로 구성된다. 데이터 플로우 그래프란, 이러한 노드들을 하나의 operation 으로 취급할 때, 엣지는 데이터 배열 또는 텐서(Tensor)로써 연결되어 최종적으로 내가 원하는 결과물을 도출 하는것이다.

결국 이러한 텐서들의 흐름을 표현하는 것이 텐서플로의 네이밍(naming) 비화이다.

### 가상환경 설치 및 설정

```
# pip 설치
$ sudo easy_install pip

# xcode 명령어 라인 도구 설치
xcode-select --install

# pyenv 설치 및 환경변수 설정
$ brew install pyenv
$ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
$ echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
$ echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
$ source ~/.bash_profile

# virtualenv 설치
$ brew install pyenv-virtualenv
$ echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bash_profile
$ source ~/.bash_profile

# 파이썬 버전 확인 후 특정 버전 설치(현재 기준 텐서플로가 지원되는 최신 버전은 3.5.6 이다)
$ pyenv install --list
$ pyenv install [python version]

# 가상 환경 생성 및 실행
$ pyenv virtualenv [python version] [environment name]
$ pyenv activate [environment name]
$ pyenv deactivate
```

### 특정 버전 파이썬이 설치되지 않을 경우

```bash
$ brew install homebrew/dupes/zlib
$ brew install readline xz
$ CFLAGS="-I$(brew --prefix openssl)/include" \
LDFLAGS="-L$(brew --prefix openssl)/lib" \
pyenv install -v 3.6.3
```

### 특정 폴더 진입시 가상환경 자동실행 설정

```bash
$ brew install autoenv
$ echo 'source /usr/local/opt/autoenv/activate.sh' >> ~/.bash_profile
$ source ~/.bash_profile

/Users/eunwoo/.zshrc:source:112: no such file or directory: /Users/eunwoo/.autoenv/activate.sh
autoenv:
autoenv: WARNING:
autoenv: This is the first time you are about to source /Users/eunwoo/Desktop/tensorflow/.env:
autoenv:
autoenv:   --- (begin contents) ---------------------------------------
autoenv:     pyenv activate tensorflow$
autoenv:
autoenv:   --- (end contents) -----------------------------------------
autoenv:
autoenv: Are you sure you want to allow this? (y/N)
```

### 가상 환경(virtualenv, pyenv) 명령어

```bash
$ pyenv install --list # 다운가능한 파이썬 버전 목록 보기
$ pyenv install [python version] # 특정 파이썬 버전 설치
$ pyenv versions # 설치된 python의 버전들 보기
$ pyenv uninstall [environment name] # 가상환경 삭제

# pyenv-virtualenv 명령어
$ pyenv virtualenv [python version] [environment name] # 가상환경 생성
$ pyenv activate [environment name] # 가상 환경 실행
$ pyenv deactivate # 가상 환경 종료
```

### 텐서플로 설치 및 확인

```bash
# 텐서플로 설치
pip3 install --upgrade tensorflow

# 설치 확인
$ python3
>>> import tensorflow as tf
>>> tf.__version__
'1.10.1'
```

## 예제 소스코드

https://github.com/hunkim/DeepLearningZeroToAll/

## 실습

### 주피터 노트북(jupyter notebook) 설치 및 실행

주피터 노트북이란 line by line으로 코드의 동작을 실행하면서 도식화된 결과를 확인하여, 머신러닝이나 빅데이터 분석의 편의성을 제공해주는 툴이다.


```bash
# 주피터 노트북 설치
$ pip3 install jupyter
$ jupyter notebook
```

### TensorFlow Hello World

```py
import tensorflow as tf
```

앞서 버전 확인에서 한 것 처럼 텐서플로 모듈을 사용하기위해 import 한다.

```py
hello = tf.constant("Hello, TensorFlow!")
```

import 된 텐서플로에 Hello, TensorFlow 라는 문자열을 상수(constant)로 설정한다. 상당히 간단해 보이지만 그래프에 문자열이 있는 노드를 생성한 것과 동일하다.

```py
sess = tf.Session()
print(sess.run(hello))
```

텐서플로는 session 을 만들고, 여기에 노드를 적용시켜 실행이 가능하다.

### Computational Graph 1

다음으로 간단한 Computational Graph를 그려보자. 2개의 노드가 주어졌을 때 이 값을 더하는 노드를 만들 것이다.

```py
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
```

우선 2개의 상수를 node1,node2로 만든다. 이 후, node3을 node1 + node2가 되도록 구현한다.

그럼 이 값들을 출력하기 위해서는 어떻게 해야할까

```py
print("node1:", node1, "node2:", node2)
print("node3:", node3)

# 결과
node1: Tensor("Const_1:0", shape=(), dtype=float32) node2: Tensor("Const_2:0", shape=(), dtype=float32)
node3:  Tensor("Add:0", shape=(), dtype=float32)
```

기대값은 상수와 두 상수의 합이었으나, 위와 같이 node를 출력할 경우에는 Tensor 객체의 정보가 출력된다.
따라서 제일 처음 예제처럼 session을 생성 후, session에서 값을 출력해야한다.
결과는 아래와 같다.

```py
sess = tf.Session()
print("sess.run(node1, node2):", sess.run([node1, node2]))
print("sess.run(node3):", sess.run(node3))

# 결과

sess.run(node1, node2):  [3.0, 4.0]
sess.run(node3):  7.0
```

### 텐서플로 머신러닝 

결과적으로 텐서플로 머신러닝은 아래의 3가지 순서로 동작한다.

1. TensorFlow operation들로 그래프를 만든다
2. sess.run을 통해 그래프를 실행시킨다
3. 그 결과 값들이 업데이트 되거나 결과값이 반환된다.

### Placeholder

앞서 만든 더하기 그래프 예제는 처음 시작할 때 상수가 고정값이며, 이를 통해 더하기 연산을 수행한다.
이와 다르게 그래프가 만들어지는 시점에서 값이 입력받기를 원한다면 Placeholder를 사용할 수 있다.

```py
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1,3], b: [2, 4]}))
```

위의 코드에서 볼 수 있듯이, feed_dict에서 값을 넘겨주게 된다. 또한 값은 한가지가 아닌 다항이나 배열의 형태로 전달이 가능하다. 

### Tensor Ranks, Shapes, and Types

- Rank: 몇 차원의 Array인지

<img src="https://user-images.githubusercontent.com/20614643/44628170-bc497a80-a975-11e8-8b47-fb6b9bde1fb5.png" width="700px"/>

- Shapes : 몇 개의 element를 가지고 있는지

<img src="https://user-images.githubusercontent.com/20614643/44628230-b6a06480-a976-11e8-9bdd-2247eba13bfb.png" width="700px"/>

- Types : 말그대로 데이터의 타입. 대부분의 경우 float32를 사용

<img src="https://user-images.githubusercontent.com/20614643/44628229-b6a06480-a976-11e8-9db8-3bdc3e00a12a.png" width="700px"/>

