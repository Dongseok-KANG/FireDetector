# FireDetector
2019 경기대학교 통계자료분석 경진대회 (최우수상 수상) 2019.05

카메라를 이용한 자동 화재 인식 및 신고 시스템

사용라이브러리 : OpenCV, Tensorflow

## 아이디어 제안 배경

작은 화재가 큰 화재로 번지는 가장 큰 이유는 늦은 화재 진압이다. 화재 진압이 지연되는 데에는 다양한 이유가 존재하고, 그 중 초기 대응이 가장 중요하다. 초기 대응을 잘 하기 위해서는 화재를 신속하게 감지하고 신고를 해야 한다. 기존의 화재경보기는 주변 환경에 따라 오작동을 일으키는 경우가 많다. 무더위가 지속되는 여름철에 열을 감지해 빈번히 화재경보기가 작동되는 것을 볼 수 있다. 이러한 문제로 화재경보기를 꺼놓는 사용자들이 생겨나고 오인출동이 발생하고 있다. 이 문제를 해결하기 위해 CNN을 이용한 화재감지 및 신고 알고리즘을 개발했다. 

## OpenCV를 이용한 1차 화재 분류

CNN을 화재 영상에 접목하기 전에, 화재 영상의 1차적인 화재 분류와 화재 구간 라벨링 작업을 위해 파이썬 OpenCV를 사용하였다.
OpenCV를 사용해, 기존의 영상의 색 속성인 RGB 속성을 HSV 속성으로 변환하는 작업을 시행하였다. RGB에서 HSV에서 변환하는 과정 중 ‘Gaussian Blur’를 이용해 영상의 노이즈를 제거한다. HSV는 특정 색상의 영역을 추출할 때 사용하며, Hue, Saturation, Value로 나눠진다. 영상을 HSV로 변환해 ‘불’의 영역을 추출해내기 위해서는 HSV의 특정값을 정해주는 작업이 필요하다. 본 분석에서는 HSV의 값을 추출해주는 트랙바를 이용해, 여러 영상들의 HSV 값의 대략적인 범위를 추출해내었다. 
그 후 HSV 변환을 통해 OpenCV가 인식한 ‘불’에 대해 라벨링을 실시한다. OpenCV가 불로 인식한 범위에 정사각형의 라벨을 씌워 보여주게 된다. 여러 영상들에서 이러한 OpenCV 작업을 실시한 결과 모든 영상에서 불을 인식을 하나, 다른 불과 유사한 HSV값을 가진 카메라 조명, 구조물에 대해서도 불로 인식한다는 문제점이 있다. 

![image](https://user-images.githubusercontent.com/52941937/80810965-e39f2300-8bff-11ea-9ee3-1054f8a895f9.png)

## CNN을 이용한 화재이미지 분류

![image](https://user-images.githubusercontent.com/52941937/80810338-92dafa80-8bfe-11ea-8e34-dd92b72b6725.png)

본 알고리즘 개발에서는 Google에서 수집한 748장의 불 이미지 중 598장을 학습 데이터(Train data), 150장을 실험 데이터(Test data)로 지정하였고, 871장의 불이 없는 이미지 중 697장을 학습 데이터(Train data), 174장을 실험 데이터(Test data)로 지정하여 학습을 진행 하였다. 1619개의 이미지를 분류하기 위해 컨볼루션(Convolution) 계층과 풀링(Pooling)계층이라는 2개의 계층이 서로 교차되어 이미지의 특징을 추출하고, 이를 3번 반복해 변화(Variation)에 영향을 받지 않는 특징을 추출하였다. 이후 2개의 완전 연결 계층(Fully connnected layer)에서 앞에서 추출된 특징을 사용해 이미지 분류를 수행한다.

학습 데이터(Train data)를 CNN의 전 방향으로 보내면서 cross-entropy cost 함수를 이용해 오차를 계산한다. 이를 다시 CNN의 역방향으로 보내면서 ‘Layer’들을 수정한다. 이런 과정을 반복하면서 CNN을 학습시킨다.

컨볼루션(Convolution) 계층과 풀링(Pooling) 계층에서 이미지의 특징을 추출 하는데 ‘Softmax’를 이용한다. 그러나 ‘Softmax’값은 0에서 1사이의 값을 갖기 때문에 ‘Layer’가 많아질수록 0으로 수렴해 이미지의 특징을 추출하는 데 문제가 생긴다. 이를 보안하기 위해 ‘Softmax’ 대신 0이하의 값은 0으로 하고, 0보다 큰 값은 그 값을 그대로 사용하는 ‘ReLu’를 사용 하여 CNN을 학습시켰다. 

![image](https://user-images.githubusercontent.com/52941937/80810741-6a073500-8bff-11ea-9ef3-bb1ec1ff351f.png)

![image](https://user-images.githubusercontent.com/52941937/80810787-83a87c80-8bff-11ea-84c3-8340aab5bc62.png)
(cost)

![image](https://user-images.githubusercontent.com/52941937/80810818-8f943e80-8bff-11ea-96f0-d314a1c52f79.png)
(accuracy)

또한 CNN학습 과정에서 발생하는 과적합(Overfitting)을 방지하기 위해 ‘Dropout’과 ‘Maxpooling’ 방법을 사용하였다. ‘Dropout’은 임의의 확률로 ‘Layer’들을 학습에서 제외시켜 과적합(Overfitting)을 방지하는 방법이고 ‘Maxpooling’은 풀링(Pooling)을 할 때 최대값을 대푯값으로 설정하는 방법이다. 결과적으로 정확도(Accuracy)가 1.0에 수렴하면서 학습이 잘 된 것을 볼 수 있다.

![image](https://user-images.githubusercontent.com/52941937/80810533-07ae3480-8bff-11ea-8327-878fde130577.png)

실험 데이터(Test data)를 이용해 CNN이 화재이미지를 잘 분류하는지 평가 하였을 때 정확도(Accuracy)는 0.94로 CNN이 새로운 이미지에 대해서도 분류를 잘 시행하는 것을 확인 하였다. 컴퓨터의 성능 문제로 많은 양의 데이터를 학습 시키지 못하고 픽셀 또한 28X28로 줄여서 학습을 시켰다. 고성능의 컴퓨터를 사용한다면 더 높은 정확도를 갖는 CNN을 만들어 화재이미지를 완벽하게 분류 할 수 있을 것으로 예상된다.

![image](https://user-images.githubusercontent.com/52941937/80810630-3d531d80-8bff-11ea-933d-886b3392d3fd.png)

(Tensorflow graph)


## 결론 : CNN 적용 및 신고

OpenCV 패키지를 이용해 라벨링(Labeling)한 화재 예상 영역을 이미지로 변환하고 CNN을 통해 화재 이미지 여부를 판단한다. 
![image](https://user-images.githubusercontent.com/52941937/80811070-206b1a00-8c00-11ea-8596-01a0dd00875b.png)

(OpenCv를 통해 판별한 화재 -1차 분류)

![image](https://user-images.githubusercontent.com/52941937/80811131-3a0c6180-8c00-11ea-8967-65b743e7ced7.png)

(CNN을 통한 2차 분류)

라벨링(Labeling)한 영역이 화재이미지 라고 판단되면 즉시 119에 문자신고를 한다.
(아이디어 측면에서 점수를 받기 위해 문자 신고 라이브러리를 사용했지만... 본 프로젝트는 OpenCv와 Tensorflow에 중점을 두고 진행하였다.)
CNN 분류기를 사용해 화재를 인식한 결과 OpenCV 단독으로 사용 했을 때 잘못 인식했던 조명, 구조물 등을 화재로 인식하지 않았다.
 
이를 통해 앞서 언급했던 화재를 조기에 막기 위한 조기신고를 충족시킬 수 있을 것으로 예상된다. 향후 CNN을 이용한 화재 감지 및 신고 시스템을 더욱 안정화 시킨다면 실제 스마트폰이나 블랙박스와 같은 카메라 기능을 가진 기기들에 적용할 수 있을 것이다. 이를 통해, 기존의 화재경보기, 감지기들이 가진 문제점을 해결해줄 수 있을 것이며, 더욱 신속하고 정확한 화재 신고 방안으로 작용할 수 있을 것으로 기대한다.
