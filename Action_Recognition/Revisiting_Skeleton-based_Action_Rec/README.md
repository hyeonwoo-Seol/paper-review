# Revisiting Skeleton-based Action Recognition

## Korean
Single Modality model인 PoseConv3D의 입력 표현은 2D heatmap을 시간축으로 쌓은 3D heatmap volume 입니다. 그리고 여러 3D-CNN을 backbone으로 사용할 수 있습니다. 3D-CNN backbone을 사용할 때 초기 down-sampling 연산을 제거합니다.

Multi Modality model인 RGBPose-Conv3D는 RGB경로와 Pose경로를 비대칭으로 구성하고 bidirectional lateral connections을 추가해서 두 경로를 빠르게 결합합니다.



## Abstract
GCN-based methods are subject to limitations in robustness, interoperability, scalability.

PoseConv3D relies on a 3D heatmap volume instead of a graph sequence.

PoseConv3D is more effective in learning spatio-temporal features, and more robust against pose estimation noises, and generalizes better in cross-dataset settings. And it can handle multiple-person scenarios without additional computation cost.

Hierarchical features can be easily integrated at early fusion stages.

## 핵심 아이디어
3D heatmap volume input


## 방법론
PoseConv3D is a 3D-CNN-based model for skeleton-based action recognition.

2D poses are better quality compared to 3D poses. And Top-down methods have superior performance on COCO-keypoints than Bottom-up. So this paper adopt 2D Top-Down pose estimators for pose extraction.

This paper find that coordinate-triplets (x, y, c) help save the majority of storage space at the cost of little performance drop.

This paper reformulates 2D pose into 3D heatmap volume. It represent a 2D pose as a heatmap of size K x H x W, where K is the number of joints, H and W are the height and width of the frame.

This paper can use three type for making heatmap.

First is using the heatmap produced by the Top-down pose estimator as the target heatmap.

Second is obtaining a joint heatmap J by composing K gaussian maps centered at every joint. This type can use when we have only coordinate-treplets (x, y, c) of skeleton joints.

Third is creating a limb heatmap L.

Finally, a 3D heatmap volume is obtained by stacking all heatmaps (J or L) along the temporal dimension, which has the size of K x T x H x W.

This paper apply two techniques to reduce the redundancy of 3D heatmap volume. First is 'subjects centered cropping'. And second is 'Uniform Samppling'.

First, this paper find the smallest bounding box that envelops all the 2D pose across frames. And Crop all frames according to the found box and resize them to the target size.

Second, this paper sampling a subset of frames due to reduce temporal dimension. This paper propose to use a uniform sampling strategy for 3D-CNN. It divide the video into n segments of equal length and randomly select one frame from each segment. Thus, it choise n frame and the heatmap volume is constructed only for these n frames.

이 논문은 Pose modality를 위한 PoseConv3D와 RGB+Pose dual-modality를 위한 RGBPose-Conv3D라는 두 가지 계열의 3D-CNN을 설계했습니다.

PoseConv3D는 세 가지 3D-CNN(C3D, SlowOnly, X3D)을 backbone으로 각각 사용했고, 3D heatmap volume을 input으로 받습니다. 이 때 CNN backbone의 down-sampling 연산은 제거됩니다.

RGBPose-Conv3D는 RGB modality와 Pose modality를 각각 처리하는 2개의 경로로 구성된 2-stream 3D-CNN 입니다. 이 두 경로는 asymmetrical입니다. pose 경로는 RGB 경로에 비해 채널 폭, 네트워크 깊이, 입력 공간 해상도가 작습니다. 그리고 두 경로 사이에 bidirectional lateral connections가 추가되어, early step에서 두 modality 사이의 feature fusion이 촉진됩니다. 그리고 Overfitting을 방지하기 위해 각 경로에 cross-entropy Loss Function을 사용합니다.


## 결론
This paper proposed PoseConv3D, a 3D-CNN based approach for skeleton-based action recognition.

PoseConv3D takes 3D heatmap volumes as input.

Because of compact 3D heatmap volume as input, PoseConv3D outperforms GCN-based approaches in both accuracy and efficiency.
