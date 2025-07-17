# VGG

## 핵심 아이디어
깊이가 깊어질 수록 대규모 인식에서 정확도가 향상된다.

작은 크기인 3 x 3 Conv filter를 여러 번 사용하여 수용 영역을 확보함과 동시에 파라미터 수를 줄인다.

학습할 때 scale jittering을 통한 데이터 증강이 일반화 성능을 향상시킨다.

테스트 시 Dense Evaluation와 Multi-crop Evaluation를 병행하면, 정확도가 향상된다.

## 방법론
![Table1](image/Table1.png)

Input Layer takes input size of 224 x 224 RGB image.

This paper subtracted the mean RGB value from each pixel. This is only pre-processing in this paper.

It use 3 x 3 filters and stride 1 and padding 1. And it use 2 x 2 max pooling with stride 2.

A stack of convolutional layers is followed by three Fully-Connected layers. The first two have 4096 channels each, the third perform 1000-way classification and thus contains 1000 channels.

The final layer is softmax layer.

All hidden layers are use ReLU as activation function.

Local Response Normalization does not improve the performance on the ILSVRC, so this paper doesn't contain LRN execpt for one.

Number of channels are increasing by a factor of 2 after each max-pooling layer.

Instead of a single 7×7 convolutional layer, three 3×3 convolutional layers were used to construct a more discriminative decision function. This also helped reduce the number of parameters.

By using 1×1 convolutional layers with ReLU, it is possible to increase the non-linearity of the decision function without affecting the size of the receptive field.


## Training
momentum을 사용하는 mini-batch gradient descent로 Multinomial Logistic Regression 목적 함수를 최적화 합니다.

batch size는 256이고, momentum은 0.9 입니다. Learning rate는 10^-2로 설정했고, validation set의 accuracy가 더 이상 향상되지 않으면 learning rate를 10으로 나누었습니다. 총 3번의 학습률 감소가 있었고, 전체 학습은 370K회(74 epochs) 입니다.

Glorot & bengio (2010)의 무작위 가중치 초기화 방법을 사용하면 pre-training 없이도 가중치를 안정적으로 초기화할 수 있습니다.

## Testing
The first FC Layer is converted to 7 x 7 Conv Layer and the last two FC Layers are converted to 1 x 1 Conv Layers.

This Fully-convolutional Net outputs class score map with the number of channels equal to the number of classes. And then class score map is spatially averaged (sum-pooled).

The final scores are obtained by averaging softmax class posteriors of original and flipped images.

## 결론
This paper evaluated very deep Convolutional Network for large scale image classification.

This paper use small size of filters 3 x 3. This improved accuracy and reduced parameters.

Representation Depth is beneficial for the classification accuracy.

훈련 이미지의 크기를 다양하게 조절하는 multi-scale training 기법이 고정된 크기의 이미지로 훈련하는 것보다 훨씬 좋은 결과를 가져온다는 것을 입증했습니다.

This paper combines the outputs of several models by averaging their soft-max class posteriors. This
improves the performance due to complementarity of the models.

Increassing depth, Convnetwork architecture achieved State-of-the-Art performance on the ImageNet.

This paper confirm the importance of depth in visual representations.
