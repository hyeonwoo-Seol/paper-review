# Three-Stream Convolutional Neural Network with Multi-task and Ensemble Learning for 3D Action Recognition

## Co-Occurrence Feature Learning with CNN
An action is associated with and characterized by the interactions and combinations of a subset of skeleton joints.

HCN's convolution operation is decomposed into two steps.

In first step, an independent convolution kernel slides on each channel of input. So the features are aggregated loccaly.

In second step, an elemnet-wise summation across channel is used. So the features are aggregated globally.

## Feature Enhancement
Bone segments provide the crucial cues to describe the human action. Because, bone segments can directly reflect the body's length and direction information

## Conclusion
This paper proposed Three-Stream CNN model, 3SCNN.

This model handles the three kinds of inputs jointly. skeleton's position, motion and bone segment.

This paper designed the coordinate adaptive module to enrich feature expression.

And it proposed a pairwise feature fusion scheme and a multi-task ensemble learning network.

These take advantage of the complementary and diverse nature among multiple features.

3SCNN shows impressive performance on NTU RGB+D dataset and outperforms other SOTA methods.
