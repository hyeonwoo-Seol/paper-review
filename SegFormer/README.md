# SegFormer paper Summary
## 1. 논문 정보
제목: SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers

저자: Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo

## 2. Abstract 요약
SegFormer는 단순하고 효율적이며 강력한 Semantic Segmentation Framework입니다. SegFormer는 Transformer와 경량의 MLP 디코더를 통합한 구조입니다.

SegFormer는 계층적으로 구성된 새로운 Transformer 인코더를 사용하여 Multiscale Feature를 출력합니다. 이 인코더는 Positional Encoding을 사용하지 않기 때문에 Test Resolution과 Training Resolution이 다를 때 발생하는 위치 코드 보간에 대한 성능 저하가 발생하지 않습니다. 그리고 SegFormer는 복잡한 디코더를 사용하지 않고 논문이 제안한 MLP 디코더를 사용합니다. 이 디코더는 서로 다른 층의 정보를 집계하여 Local Attention과 Global Attention을 모두 결합하여 강력한 표현력을 제공합니다.

이 논문에서는 SegFormer-B0 부터 B5까지 모델을 확장했으며, SegFormer-B4는 ADE20K에서 64M 파라미터로 50.3% mIoU를 달성함과 동시에 이전의 최고 방법보다 모델의 크기가 5배 작고 성능이 2.2% 더 우수합니다. SegFormer-B5는 Cityscapes 검증 세트에서 84% mIoU를 기록하고 Cityscapes-C에서 Zero-shot 강건성을 보여주었습니다.

## 3. 문제 정의 및 동기
Pyramid VisionTrnasformer와 Swin Transformer, Twins 등의 최신 방법들은 주로 Transformer의 인코더 설계에 집중했고, 더 나은 성능을 위한 디코더의 설계는 상대적으로 간과했습니다. 이에 논문의 저자는 기존 방법들과 달리 인코더와 디코더를 모두 새롭게 설계하고자 합니다.

## 4. 핵심 아이디어
Positional Encoding이 없는 새로운 hierarchical Transformer 인코더와 복잡하고 계산량이 많은 모듈 없이도 강력한 표현력을 제공하는 경량 All-MLP 디코더 설계를 통해 세 가지 공개 Semantic Segmentation 데이터셋에서 효율성, 정확도, 강건성의 State-of-the-Art를 달성했습니다.

Positional Encoding을 제외함으로써 Test와 Training의 이미지 해상도가 다를 경우 위치 코드 보간을 하지 않도록 했습니다. 이를 통해 성능 저하 없이 임의의 테스트 해상도에 유연하게 대처할 수 있습니다. 그리고 hierarchical 구조를 통해 고해상도의 정밀한 특징과 저해상도의 거친 특징을 모두 생성할 수 있습니다.

경량 MLP 디코더는 Transformer의 하위 계층 Attention이 Local에 집중하고 상위 계층 Attention이 Global에 집중하는 것을 바꿨습니다. 경량 MLP 디코더는 Local과 Global Attnetion을 모두 통합할 수 있게 해서, 단순하고 직관적인 디코더 구조로 강력한 표현력을 얻었습니다.

## 5. 방법론
Figure2는 SegFormer의 구조를 보여줍니다.

![Figure2: The proposed SegFormer Framework](image/Figure2.png)

### Hierarchical Transformer Encoder
아키텍처가 동일하지만 크기만 다른 Mix Transformer-B0 부터 B5까지 설계했습니다. Mix Transformer에는 PVT의 계층적 아키텍처 Efficient Self-Attention 모듈 위에 Overlapped Patch Merging과 Positiona-Encoding-Free Design 등을 추가했습니다. 인코더에는 입력 이미지를 기반으로 multi-level multi-scale 특징들을 생성합니다. 이러한 특징들은 고해상도의 거친 특징과 저해상도의 세밀한 특징을 함께 제공하기 때문에 Semantic Segmentation 성능을 향상시킵니다. 각 Transformer Block 안에 Overlap Patch Merging을 수행해서 해상도가 H/2^(i+1) x W/2^(i+1) x C_i 인 Hierarchical Feature map을 생성합니다.

계층적 특징 표현은 높은 해상도의 특징으로부터 긴 시퀀스가 생성될 때 self-Attention의 복잡도가 제곱 형태가 되는 문제가 있어서, Transformer 불록 안의 Multi-head Self-Attention에 Sequence Reduction 기법을 적용합니다. 아래 eq2를 사용하여 시퀀스 길이를 줄입니다. 첫 번째 식은 K를 N/R x (C ⋅ R) 형태로 변형하고, 두 번째 식은 (C ⋅ R) 차원의 텐서를 입력으로 받아서 C 차원의 텐서를 출력하는 선형 계층을 나타냅니다. 새로운 K는 N/R x C 의 차원을 가지게 되고, 이로 인해 Self-Attention의 복잡도 O(N^2)가 Efficient Self-Attention의 복잡도 O(N^2/R)로 줄어들게 됩니다. 논문의 저자는 실험을 통해 Stage1 ~ 4까지의 R값을 {64, 16, 4, 1}로 설정했습니다.

![eq2](image/eq2.png)

이웃한 패치 사이의 Local Continuity를 유지하기 위해, Overlapping Patch Merging을 사용합니다. 이 때 K(patch size), S(stride between two adjacent patches), P(padding size)를 정의하고, K=7 / S=4 / P=3 과 K=3 / S=2 / P=1로 설정하고 이 값으로 Overlapping Patch Merging을 수행했습니다. 이를 통해 non-overlapping과 동일한 크기의 특징을 생성합니다.

Test와 Training 이미지의 해상도 불일치는 Semantic Segmentation에서 자주 발생하고, 이로 인해 Positional Encoding을 보간(interpolate)해야 하고, 이는 성능 하락으로 이어집니다. SegFormer는 Positional Encoding을 제거하고 Mix-FFN을 도입했습니다. FFN에서 3x3 Conv를 직접 사용해서 Zero Padding이 위치 정보를 누락시키는 효과를 고려합니다. Mix-FFN의 수식은 ![eq3](image/eq3.png) 입니다. 이 수식에서 x_in은 Self-Attention 모듈에서 나온 Feature이고, Mix-FFN은 FFN에 3x3 Conv 와 MLP를 혼합하여 사용합니다. 그리고 파라미터 수를 줄이고 효율성을 높이기 위해 Depth-wise Convolution을 사용합니다.

### Lightweight All-MLP Decoder


## 6. 실험 결과

## 7. 결론

## 8. 느낀점
