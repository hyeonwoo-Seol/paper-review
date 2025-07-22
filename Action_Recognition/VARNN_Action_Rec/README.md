# View Adaptive Recuurent Neural Networks for High Performance Human Action Recognition from Skeleton Data

## RNN and LSTM Overview
![Figure3](image/Figure3.png)
Standard RNN faces the vanishing gradient effect, which is not very capable of handling long-term dependencies.

In LSTM, the removal of the previous information or adition of the current information to the cell state are regulated with linear interactions by the forget gate and the input gate.

## View Adaptation Model using LSTM
![Figure2](image/Figure2.png)

Observation Coordiate System obtained from Global Coordinate System by translation and rotation.

View Adaptation module is end-to-end LSTM network.

View Adaption Subnetwork determine the observation viewpoint automatically.

View Adaptation Subnetwork is followed by Main LSTM Network that learning the temporal dynamics and performing the feature abstractions from the view-regulated skeleton data for action recognition.

Two branches of LSTM takes skelton V_t as input. This LSTM subnetworks are utilized to learn the rotation parameters to obtain the rotation matrix, and to learn the translation vector. This parameters are corresponded to the global coordinate system.

The branch of rotation subnetwork for learning rotation parameters consists of an LSTM layer and a FC layer.

The branch of translation subnetwork for learning translation parameters consists of an LSTM layer and a FC layer.

The Main LSTM network consists of three LSTM layers which is followed by one FC layer with softmax classifier.

With end-to-end training, the view adaptation model is guided to select the suitable observation viewpoints for enhancing recognition accuracy.

## Conclusion
View Adaptive Network is End-to-End Model for human action recognition.

VA Network is capable of regulating the observation viewpoint to the suitable ones by itself.

It overcomes the limitations of the human defined pre-processing approaches.

The proposed model improves recognition performance on three benchmark datasets and achieve SOTA results.
