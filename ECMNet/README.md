# ECMNet paper summary
## 논문 정보
제목: ECMNet:Lightweight Semantic Segmentation with Efficient CNN-Mamba Network

저자: 
## Abstract 요약
이 논문은 Efficient CNN-Mamba Network for Semantic Segmentation이라고 불리는 경량화된 네트워크를 제안합니다.

ECMNet은 Capsule-based Framework 내에서 CNN과 Mamba를 효과적으로 결합하여 이 둘의 상호 보완적인 약점을 해결합니다.

Enhanced Dual Attention Block (EDAB)를 설계했고, 특징 표현력 향상을 위해 Multi-Scale Attention Unit (MSAU)를 사용해서 다중 스케일 특징 통합과 공간적 통합과 채널 통합을 수행합니다.

Mamba 기반 특징 융합 모듈 (Feature Fusion Module)을 도입해서 다양한 수준의 특징을 효과적으로 융합함으로써 분할 정확도를 크게 향상시켰습니다.

## 문제 정의 및 동기
초기 Semantic Segmentation은 CNN을 사용했지만, CNN은 장거리 의존성을 포착하는 데 한계가 있었습니다.

이후에 나온 Transformer는 Semantic Segmentation에 효과적인 Global Context Modeling을 가능하게 했습니다. 이를 통해 전역 특징을 추출하고 복잡한 장면 데이터셋에서 기존의 CNN 모델들보다 뛰어난 성능을 보여줬습니다.

SegFormer는 Hierarchial Transformer Encoder와 경량 MLP Decoder를 결합해서 다중 스케일 특징 융합을 최적화했습니다. 그러나 Transfomrer는 연산 복잡도가 이미지 해상도의 제곱에 비례하기 때문에 고해상도 처리 시 연산 부담이 크고, 지역 정보에 대한 민감도가 부족하다는 한계가 있습니다.

위와 반대로 CNN 인코더와 Transformer 디코더를 결합한 모델들도 있었지만, 여전히 Self-Attention이 고해상도 이미지에서 장거리 시각 의존성을 처리할 때 속도와 메모리 사용량 측면에서 한게가 있었씁니다.

Mamba는 선형 복잡도의 효율적인 시퀀스 모델링을 통해 고해상도 이미지 처리에서 큰 가능성을 보여줬고, Vision Mamba는 다양한 컴퓨터비전 과업에서 뛰어난 성과를 입증했습니다.

게다가 제한된 계산 자원 및 모바일 디바이스에서 적용 가능한 Ligthweight Semantic Segmentatoin인 LEDNet, CFPNet, LETNet 등도 개발되었습니다.

위 두 방식인 Mamba와 Lightweight Semantic Segmenation을 기반으로 CNN-Mamba Hybrid Network인 ECMNet을 제안합니다.

## 핵심 아이디어
### U자형 CNN 인코더-디코더 구조 (Backbone)
이 구조를 통해 세부족인 공간 표현을 위한 Localized Features를 추출합니다.

### Feature Fusion Module (FFN) + SS2D Block
State Space Model (SSM)을 활용해서 복잡한 공간 정보와 장거리 의존성을 포착하고 Global Feature Representation 과 계산 복잡도를 최적화합니다.

### Enhanced Dual-Attention Block (EDAB)
서로 다른 수준의 특징 정보를 효과적으로 포착하면서 네트워크 파라미터 수를 최소화하도록 설계된 블록입니다.

### Multi-Scale Attention Unit (MSAU)
저수준 Spatial 정보와 고수준 Semantic 정보에 집중해서 더 높은 품질의 분할 결과를 생성합니다.

## 방법론
![Figure1](image/Figure1.png)

Figure1을 보면, CNN 인코더와 디코더는 EDAB로 개선된 모습이 보이고, 인코더와 디코더를 연결해주는 Mamba 기반의 Feature Fusion Mdoel이 보입니다.

그리고 3개의 Multi-Scale Attention Unit이 인코더와 디코더 사이에서 Long Skip Connection 역할을 담당합니다.

### 인코더 - 디코더
입력 특징은 1x1 Conv를 활용한 Bottleneck 구조를 통과해서 채널 수를 절반으로 줄입니다. 이를 통해 연산 복잡도와 파라미터 수를 크게 감소시킵니다.

1x1 Conv를 사용하지 않고, 3x1 과 1x3 Conv를 사용할 수도 있습니다. 이렇게 2개로 분리시켜두면 더 넓은 Recpetive Field를 확보하고 더 넓은 Contextual Feature를 포착할 수 있으며, 모델의 파라미터 수와 계산량도 타협할 수 있습니다.

EDAB는 2개의 결로를 가지고 있고, 각 경로는 Local 특징과 Global 특징을 포착합니다.

하나의 경로는 Decompose Convolution을 통해 Local이고 Short-distance 특징을 처리하며, 다른 하나의 경로는 Atrous Convolution을 통해 GLobal feature Integration을 수행합니다.

이 두 경로에 각각 Channel Attention 과 Dual-Direction Attention을 적용시켜서 다차원 특징 정보를 학습하고 특징 표현력을 향상시켰습니다. 왜냐하면 상당수의 유용한 정보들은 채널 차원에 포함되고, 공간적 특징 정보는 성능향상과 잡은 간섭 억제에 핵심적인 역할을 하기 때문입니다.

## 실험 결과

## 결론

## 느낀점
