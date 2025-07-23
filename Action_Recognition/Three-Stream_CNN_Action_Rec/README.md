# Three-Stream Convolutional Neural Network with Multi-task and Ensemble Learning for 3D Action Recognition

## Co-Occurrence Feature Learning with CNN

An action is associated with and characterized by the interactions and combinations of a subset of skeleton joints.

HCN's convolution operation is decomposed into two steps.

In first step, an independent convolution kernel slides on each channel of input. So the features are aggregated loccaly.

In second step, an elemnet-wise summation across channel is used. So the features are aggregated globally.

![Figure2](image/Figure2.png)

## Feature Enhancement

Bone segments provide the crucial cues to describe the human action. Because, bone segments can directly reflect the body's length and direction information.

Coordinate Adaptive Module은 서로 다른 시점에서 본 같은 skeleton sequence를 회전 변환을 통해 여러 관점의 정보를 얻음으로써 행동 표현의 풍부함과 식별력을 높이기 위한 방법이다.

이 모듈에는 Multi-Coordinate Transformation, Point-Level Convolution, Rotation Matrix Learning으로 구성되어 있다.

Multi-Coordinate Transformation은 여러 회전 행렬로 다양한 시점의 시퀀스를 생성한다.

Point-Level Convolution에는 1x1 Convolution과 1x3 Convolution이 있는데, 1x1은 여러 좌표계로부터 나온 L개의 시퀀스를 채널 차원에서 적응적으로 결합하고, 1x3은 시간축 상의 Point-Level feature를 추출한다. 이 두 계층을 연속적으로 적용해서 회전 변환된 정보들을 효과적으로 융합한다.

Rotation Matrix Learning은 회전행렬을 학습 가능한 파라미터로 두고, L개의 FC layer를 통해 각각의 회전행렬을 데이터로부터 학습시킨다.

## Pairwise Feature Fusion Learning

Position, Motion, and Bone segment features are extracted independently.

Pairwise Feature Fusion (PFF) consists of two procedures which are pairing and fusing for each feature.

In pairing, any two of three features can be made a pair by concatenating operation. Then, there are two alternative fusion architectures in fusing. Two fusion architectures are shared fusion and split fusion.

In split fusion, each pairwise feature possesses exclusive fusion block to learn fusion pattern.

In shared fusion, learning the fusion pattern of the three pairs uses one shared block.

## Multi-task and Ensemble Learning Network

Three features after pairwise fusion learning are sent to their own classifier. And three-stream model predicts three probability vectors for each action.

This vectors are optimized using multi-task learning problem with cross-entropy loss and each classifier produces a loss component.

During inference, this paper refer to the main idea of ensemble learning.

Last FC layer fo each classifier is jointly used to make decision for human action recognition.

For mitigating the high level of noise to make ambiguous classification, this paper choose the sum of rule to joint these feature vectors. This is called Softmax...

## Conclusion

This paper proposed Three-Stream CNN model, 3SCNN.

This model handles the three kinds of inputs jointly. skeleton's position, motion and bone segment.

This paper designed the coordinate adaptive module to enrich feature expression.

And it proposed a pairwise feature fusion scheme and a multi-task ensemble learning network.

These take advantage of the complementary and diverse nature among multiple features.

3SCNN shows impressive performance on NTU RGB+D dataset and outperforms other SOTA methods.
