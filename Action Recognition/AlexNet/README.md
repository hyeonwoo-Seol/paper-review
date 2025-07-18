# AlexNet

## Korean
ReLU 활성화함수를 사용해서 tanh에 비해 학습속도를 빠르게 했다.

다중 GPU를 활용해서 병렬로 AlexNet을 훈련시켰다.

풀링 윈도우가 겹치도록 stride를 윈도우 크기보다 작게 설정해서 error rate를 감소시켰다.

Local Response Normalization을 사용해서 특정 뉴런의 출력을 주변 뉴런들의 출력값으로 정규화한다.

Overfitting을 방지하기 위해, 이미지 변환 및 반전과 RGB 채널 강도 변경을 수행했다.

Dropout 기법을 사용해서 Overfitting을 방지한다. 약 50%의 뉴런에 Dropout 기법을 적용시켰다.

5개의 합성곱 계층과 3개의 완전연결 계층으로 이루어져있고, 총 6000만개의 파라미터와 65만개의 뉴런을 가진다. 

확률적 경사 하강법을 사용했고, 배치 크기는 128, 모멘텀은 0.9, 가중치 감쇠는 0.0005로 설정했다. 학습률은 Validation Error가 개선되지 않을 때마다 10으로 나누었고, 총 3번 나누었다.

중간의 합성곱 계층 중 하나라도 제거되면 성능이 저하되는 것을 통해 이 논문은 depth가 중요하다는 것을 보여주었다.

## 핵심 아이디어
ReLU

Multiple GPUs

Local Response Normalization

Overlapping Pooling

Data Augmentation

Dropout

## 방법론

### DataSet
ImageNet is a dataset of over 15M labeled high resolution images with 22,000 categories.

This paper down-sampled the images to a fixed resolution of 256 x 256. This paper only pre-process the image that subtracting the mean activity over the training set from each pixel.

### Architecture
![Figure2](image/Figure2.png)

It contains eight learned layers that consist of five convolutional layers and three fully-connected layers.

It has 1000-way softmax. That is, the model is capable of classifying 1,000 categories.

This network maximizes the multinomial logistic regression objective.

The kernel of Second, Fourth, Fifth Conv Layers are connected only to those kernel map in the previous layer which reside on same GPU. But the kerenl of Thrid Conv Layers are connected to all neurons in the previous layer.

Response Normalization layers follow teh First and Second Conv Layers.

Max-pooling Layers follow both response normalization layers and Fifth Conv layers.

ReLU is applied to the output of every conv and fully-connected layer.

Input image size of First Conv layer is 224 x 224 x 3. This Conv Layer filters input images with 96 kernels of size 11 x 11 x 3 with a stride of 4 pixels.

Second Conv Layer takes input which is output of the First Conv Layer. And then Second Conv Layer filters it with 256 kernels of size 5 x 5 x 48.

Third and Fourth and Fifth Conv layers are connected to one another without any interventing pooling or normalization layers.

Third Conv Layer has 384 kernels of size 3 x 3 x 256 and Fourth has 384 kernels of size 3 x 3 x 192 and Fifth has 256 kernels of size 3 x 3 x 192.

The Fully-connected layers have 4096 neurons each.

#### ReLU
It contains ReLU Nonlinearity. Saturating Nonlinearities such as tanh(x) or f(x) = (1+e^-x)^-1 are much slower than the non-saturating nonlinearity.

DCNN with ReLU train several times faster than tanh. This is demonstrated in Figure 1.

#### Multiple GPU
This paper put half of the kernels on each GPU.

The two-GPU network takes slightly less time to train than the one-GPU.

#### Local Response Normalization
Local Response Normalization aids generalization.

This is reduce Top-1 and Top-5 error rate by 1.4% and 1.2%.

#### Overlapping Pooling
Pooling layers in CNN summarize outputs from nearby neuron groups within a kernel map. Overlapping Pooling reduces Top-1 and Top-5 error rate by 0.4% and 0.3%, as ccompared with the non-overlapping pooling.

### Reduce Overfitting
#### Data Augmentation
The Transformed images are generated in Python Code on the CPU while GPU is training on the previous batch of images.

Data augmentation consists of generating images translations and horizontal reflections. This paper does this by extracting random 224 x 224 patches from the 256 x 256 images and training networks on these extracted patches.

Additionally, Data augmentation consists of altering the intensites of the RGB channels in training images. This paper performs PCA on the set of RGB pixel value throughout the ImageNet training set. To each training image, this paper adds multiples of the found principal components with magnitudes proportional to the corresponding eigenvalues times a random variable drawn from a Gaussian with mean zero and standard deviation 0.1.

This augmentation method approximates the property that the identity of an object is preserved under variations in lighting intensity and color. Unlike simple brightness adjustment, this method enhances diversity while preserving the overall color statistics of the image.

#### Dropout
Dropout is a technique that randomly disables some neurons during training to prevent overfitting and produce a more generalized model.

50% of neurons are disabled during training. At test time, network uses all the neurons that multiply their outputs by 0.5.

### Details of learning
This network uses stochastic gradient descent with a batch size of 128 examples, momentum of 0.9, weight decay of 0.0005.

The Update rule for weight, SGD equation is ![eq1](image/eq1.png)

This paper initialized the weights from a zero-mean Gaussian distribution with standard deviation 0.01.

And it initialized the biases in the Second, Fourth, Fifth Conv layer, Fully-Connected layers with constant 1. And then remaining layers are initialized with the contant 0.

All layers use same learning rate. and it is initialized at 0.01. If the validation error rate stopped improving, learning rate is divided by 10. In this paper, the learning rate reduced three times.

## 결론
On ILSVRC-2010, this paper achieves Top-1 test set error rates of 37.5% and Top-5 test set error rates of 17.0%.

On ILSVRC-2012, this paper achieves Top-5 error rate of 18.2%.

Averaging predictions of two CNN that pre-trained on entire Fall 2011 and five CNN gives an error rate of 15.3%.

On Fall 2009, this paper achieves Top1 error rates of 67.4% and Top-5 error rates of 40.9%.

Two images that have feature activation vectors with a small Euclidean separation are similar. And then Neural Network consider this two images to be similar.

This Euclidean distance between two 4096-dimenstional real-value vectors are inefficient, but short binary codes which compressed these vectors are efficient.

Paper's network demonstrates that depth is important for achieving high performance.
