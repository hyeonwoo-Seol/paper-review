# VisionTransformer paper Summary
## 1. 논문 정보
- 제목: Vision Transformer: A Review of Architecture, Applications, and Future Directions
- 저자: Abdelhafid Berroukham, Khalid Housni, Mohammed Lahraichi

## 2. Abstract 요약
- 컴퓨터 비전(Vision) 분야는 인간의 시각 능력을 모방하여 이미지를 이해하고 해석하는 기술을 사용합니다. 대표적인 Vision 과업에는 이미지 분류(Image Classification), 객체 탐지(Object Detection), 인스턴스 분할(Instance Segmentation), 자세 추정(Pose Estimation), 이미지 생성(Image Generation), 그리고 객체 추적(Object Tracking) 등이 있습니다.
- 이러한 과업들의 공통된 목표는 복잡한 시각 정보를 정밀하게 이해하는 것입니다. 예를 들어, ‘의자에 앉아 있는 사람’을 정확히 이해하기 위해서는 사람, 의자, 그리고 이 둘 사이의 관계(앉은 동작)를 동시에 해석해야 합니다. 따라서 전체 이미지를 고려하여 의미를 파악하는 전역 문맥(Global Context)의 이해가 매우 중요합니다.
- 기존에는 이러한 Vision 과업를 처리하기 위해 합성곱 신경망(CNN, Convolutional Neural Networks) 기반의 모델이 주로 사용됐습니다. CNN은 이미지의 공간 정보를 효과적으로 처리할 수 있지만, 고정된 수용 영역(Fixed Receptive Field)을 사용하여 전역 정보를 포착하기 어렵고, 서로 다른 피처(Feature) 간의 관계를 명시적으로 모델링하기 어렵다는 한계가 존재합니다.
- 이러한 한계를 극복하기 위해, 최근에는 Transformer 기반 아키텍처가 주목받고 있습니다. 원래 Transformer는 자연어 처리(NLP)에서 개발됐으나, 이를 이미지 처리에 적용하기 위해 Vision Transformer를 연구했습니다.

## 3. 문제 정의 및 동기
### CNN 한계점 - Fixed Receptive Field 문제
- CNN(Convolutional Neural Network)은 작은 필터(Kernel)를 반복적으로 적용하여 이미지를 처리합니다. Receptive Field란, 특정 뉴런의 출력이 입력 이미지의 어느 영역에 반응한 것인지를 나타내는 개념입니다. 일반적으로 CNN의 계층이 깊어질수록 Receptive Field는 넓어지지만, 그럼에도 CNN은 국소적인(Local) 정보를 중심으로 처리하는 구조입니다.
- 이 때문에 CNN은 이미지의 일부 정보에는 민감하지만, 전체적인 전역 문맥(Global Context)이나 멀리 떨어진 요소 간의 상호작용(Long-range dependencies)을 포착하는 데 한계가 있습니다. 이를 보완하기 위해 CNN의 깊이를 늘리는 방식이 사용되기도 하지만, 깊어질수록 연산 비용이 증가하고, 멀리 떨어진 정보는 네트워크를 거치면서 점차 희미해지기 때문에 근본적인 해결책이 되지 않습니다.
- 결과적으로, 사람과 배경, 혹은 사물 간의 복잡한 관계를 동시에 고려해야 하는 고차원적인 Vision Task에서는 CNN이 불리합니다.

### CNN 한계점 - Feature 사이의 관계를 명시적으로 모델링하지 못하는 문제
- CNN은 이미지의 작은 영역을 기준으로 특징(Feature)을 추출하며, 낮은 층에서는 모서리나 색상, 중간 층에서는 윤곽, 그리고 높은 층에서는 추상적인 형태를 점진적으로 학습합니다. 하지만 이러한 구조는 각 Feature 간의 관계를 명시적으로 모델링하지 못하는 한계가 있습니다. 예를 들어, 한 이미지 안에 사람과 의자가 존재할 경우, CNN은 이 두 객체를 각각 인식할 수는 있지만, “사람이 의자에 앉아 있다”라는 관계성은 명확하게 이해하지 못합니다.
- 이는 객체 간의 의미적 연결과 전역적인 구조를 파악해야 하는 경우에 큰 단점이 됩니다. 이러한 한계를 극복하기 위해 등장한 것이 Self-Attention 기반의 Transformer 구조입니다. Self-Attention은 입력의 모든 요소를 서로 비교하면서 장거리 의존성과 관계성을 모델링할 수 있어, Vision Task에서 활용도가 높아지고 있습니다.

## 4. 핵심 아이디어
### Self-Attention
- Transformer 구조의 핵심 아이디어는 Self-Attention 메커니즘입니다. Self-Attention은 입력 데이터의 서로 다른 위치들 간 관계를 직접적으로 계산하여, 맥락을 더 효과적으로 포착합니다.
- CNN은 작은 필터를 반복적으로 적용하면서 지역(Local) 정보를 중심으로 특징을 추출합니다. 반면에 Vision Transformer(ViT)는 입력 전체를 동시에 바라보며, 서로 다른 위치들 사이의 관련성을 Self-Attention Layer를 통해 연속적으로 표현합니다.
- Self-Attention을 쉽게 말해 내가 지금 보고있는 위치가 다른 위치들과 얼마나 관련이 있는지를 계산하는 메커니즘입니다.
- 이때, 각 위치는 입력 전체의 다른 모든 위치들과의 관련도(Attention Score)를 계산하며, 이 관련도에 따라 다른 위치의 정보들을 가중합하여 새로운 특징 벡터를 생성합니다.
- 즉, 한 위치의 최종 벡터는 다른 위치들의 정보 중 어떤 것을 얼마나 참고할지를 스스로 결정하여 만들어집니다.
- 이를 회의에 비유하면 더 쉽게 이해하게 됩니다. CNN은 자신 주변 몇 명의 이야기만 듣고 결정을 내리는 구조라면, Self-Attention은 회의 참석자 전원의 의견을 모두 듣고, 누가 더 중요한 이야기를 하는지 판단한 후, 그 의견을 반영해 결정을 내리는 방식입니다.
- 이러한 Self-Attention 메커니즘 덕분에 ViT는 전역 문맥 정보(Global Context)를 자연스럽게 반영할 수 있으며, CNN이 처리하기 어려운 장거리 의존성(Long-range Dependency) 문제도 효과적으로 해결할 수 있습니다.

### Positional Encoding
- Vision Transformer에서는 입력 이미지를 일정한 크기의 패치(Patch) 단위로 나눈 뒤, 각 패치를 벡터화하여 Self-Attention Layer에 입력합니다.
- 하지만 Transformer 구조는 원래 자연어 처리(NLP)를 위해 설계됐고, 입력 토큰들의 순서를 자동으로 고려하지 않는 구조입니다. 이러한 Transformer를 Vision Task에 그대로 적용시키면 위치나 순서에 대한 정보가 기본적으로 포함되어 있지 않기 때문에, 이미지에서 각 패치가 어디에 위치해 있는지를 모델이 알 수 없는 상황이 됩니다.
- 이를 해결하기 위해 사용하는 기법이 바로 Positional Encoding입니다. Positional Encod-ing은 각 패치의 위치 정보를 담은 벡터를, 해당 패치의 특징 벡터에 더해주는 방식으로 이루어집니다. 즉, 단순히 패치의 내용뿐만 아니라 이 패치가 이미지의 어디에 있는지에 대한 정보를 함께 제공함으로써, 모델이 패치 간의 공간적 관계(Spatial Relationship)를 인식할 수 있도록 돕습니다.

## 5. 방법론
### Vision Transformer Architecture
- Vision Transformer의 아키텍처는 기본적으로 여러 개의 Transformer 블록으로 구성되며, 각 블록은 Multi-head Self-Attention과 Feedforward Neural Network로 이루어져 있습니다.
- Multi-head Self-Attention은 여러 개의 Attention Head를 병렬로 적용하여, 입력의 서로 다른 부분 간의 다양한 관계를 학습할 수 있게 합니다. 각 Head는 서로 다른 정보에 집중함으로써, 입력 전체에 대한 맥락을 동시에 학습할 수 있습니다. 또한 이러한 구조는 표현력을 높이면서도 전체 파라미터 수나 연산 비용을 크게 증가시키지 않기 때문에, 효율성과 성능 면에서 장점을 가집니다.

### Input Image Preporcessing
- ViT는 입력 이미지를 먼저 고정 크기의 패치(Patch)로 분할합니다. 이후 각 패치는 Flatten 과정을 거쳐 1차원 벡터로 펼쳐지고, 선형 임베딩(Linear Embedding)을 통해 고정된 차원의 벡터로 변환됩니다. Transformer 구조는 NLP 분야에서 개발됐기 때문에, 입력 데이터를 고정된 차원의 벡터 형태로 통일해서 입력받아야 합니다. 이미지는 공간 구조를 갖고 있지만, 이를 Transformer가 처리하기 위해서는 각 패치를 동일한 크기의 벡터로 변환하는 과정이 필요합니다. 그 이유는 이미지마다 패치의 개수가 다를 수 있고, 이를 통일된 형태로 맞추지 않으면 Transformer의 입력 구조에 적합하지 않기 때문입니다.

### Positional Encoding
- Transformer는 순서 개념이 없기 때문에, 각 패치의 공간적 위치 정보를 추가적으로 제공해야 합니다. 이를 위해 학습 가능한 위치 임베딩(Positional Embedding)을 각 패치 벡터에 더해주며, 이 과정을 통해 모델은 패치 간의 위치 관계를 학습할 수 있습니다.

### Transformer Encoder, MLP Head
- 이렇게 생성된 입력은 여러 층의 Transformer Encoder를 통과하게 됩니다. Encoder는 반복적으로 Self-Attention과 Feedforward Network를 적용하여, 패치 간의 복합적인 상호작용을 학습합니다. 모든 패치에 걸쳐 생성된 출력은 Pooling 과정을 통해 하나의 통합 벡터로 요약되며, 이 벡터는 MLP Head로 전달됩니다. MLP Head는 두 개의 Fully Connected Layer로 구성되어 있으며, 최종적으로 분류 클래스 수만큼의 Logits을 출력합니다. Logits은 각 클래스에 대한 예측 점수를 나타내며, 소프트맥스(Softmax)를 통해 확률로 변환됩니다.

## 6. 실험 결과

## 7. 결론

## 8. 느낀점
- 현재 LLM 성능 향상을 목적으로 하는 연구가 많이 진행되고 있는데, 이 구조를 잘 수정하여 Computer Vision에도 적용시킨다면 Computer Vision도 성능 향상을 얻을 수 있을 것 같다고 느꼈습니다.
