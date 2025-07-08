# 논문을 읽고 Pytorch로 코드 작성해보기

## 성능 최적화
직접 만든 코드에서 최적의 배치 사이즈를 찾기 위해, 4, 8, 10 순서대로 늘려서 8GB VRAM에서 배치사이즈가 10일때 메모리를 최대로 활용한다는 것을 확인했습니다.

총 14 Epoch를 돌렸고, Validation은 25 -> 117 -> 8 -> 3.5 -> 2.6 -> 5.1 -> 2.9 -> 2.2 -> 1.4 -> 1.3 -> 1.9 -> 1.9 -> 1.9 -> 2.3 으로 나왔습니다.

처음에는 Validation Loss가 점점 줄어들다가 후반에 갈 수록 오히려 올라가는 모습을 보입니다.

이에 다음과 같은 과정을 진행하고자 합니다.

1. ReduceLROnPlateau 스케줄러를 사용하여 검증 손실이 정체될 때 자동으로 학습률을 낮추기
2. 옵티마이저에 weight decay를 추가하여 모델의 가중치가 너무 커지지 않도록 규제하기
3. 데이터 증강하기
4. 검증 데이터셋 비율을 더 늘리기


1번과 2번 방법까지 진행했을 때는 눈에 띄게 Validation Loss가 줄어들지 않았습니다. object detection에서는 증강을 할 때 Albumentations로 진행합니다. 따라서 훈련 데이터만을 Albumentations로 증강시켰더니 Training Loss와 Validation Loss가 매 Epoch마다 줄어들기 시작했습니다.

최저 loss인 10 epoch부터 다시 훈련을 시작해서, 1.3 -> 0.7 -> 0.58 -> 0.54 -> 0.50 -> 0.43 -> 0.42 -> 0.39 -> 0.37 -> 0.34 -> 0.34 -> 0.31 -> 0.32 -> 0.33 -> 0.31 -> 0.28 처럼 더 내려가기 시작했습니다.
하지만 0.3 근처에서 더 이상 진전되지 않는 모습을 보이고 있습니다.

## 코드 작성
> we use swish activation

논문에서는 swish activation을 사용했다고 적혀 있습니다. 따라서 Swish Activation을 구현합니다.
```
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
```
>  we use depthwise separable convolution [5, 34] for feature fusion, and add batch normalization and activation after each convolution.

논문에서 Depthwise Separable convolution을 사용했다고 적혀 있습니다. 이에 대한 코드는 다음과 같습니다.
```
class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias = False)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = True)
        self.bn = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 1e-3)
        self.swish = Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.swish(x)
        return x
```
이제 BiFPN 층을 구현해야 합니다.
