# Revisiting Skeleton-based Action Recognition

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

---
3.3 3D CNN for skeleton base Action Rec 부터 읽기

---

## 결론
This paper proposed PoseConv3D, a 3D-CNN based approach for skeleton-based action recognition.

PoseConv3D takes 3D heatmap volumes as input.

Because of compact 3D heatmap volume as input, PoseConv3D outperforms GCN-based approaches in both accuracy and efficiency.
