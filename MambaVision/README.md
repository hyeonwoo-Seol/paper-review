## 1. 논문 정보
제목: MambaVision: A Hybrid Mamba-Transformer Vision Backbone
저자: Ali Hatamizadeh Jan Kautz

## 2. Abstract 요약
Vision 응용 분야에 특화된 새로운 Hybrid Mamba Transformer Backbone인 MambaVision을 제안합니다.
Mamba를 시각적 특징을 처리할 수 있게 재설계했으며, Mamba에 Self-Attention을 추가하여 Mamba와 Vision Transformer를 결합하여 성능 향상을 확인했습니다.
ImageNet-1K 데이터셋의 이미지 분류 실험에서, MambaVision 계열 모델들은 Top-1 정확도와 이미지 처리량 측면에서 State-of-the-Art 성능을 달성했습니다.
그리고 객체 검출, 인스턴스 분할, 의미 분할 등의 DownStream 과제에서 유사한 크기의 다른 Backbone 보다 우수한 성능을 보였습니다.

## 3. 문제 정의 및 동기
###Mamba
Mamba는 Transformer의 대안으로 제안된 Selective State Space Model(SSM) 계열의 대표적 아키텍처입니다. Mamba는 기존 Transformer가 갖는 O(N^2 ) 연산 및 메모리 부담을 선형 시간 복잡도(Linear Time Complexity)로 경감하면서도 content based 정보 선택 기능을 도입해 언어 모델링, 오디오, 게놈 데이터 등 다양한 시퀀스 작업에서 Transformer와 동등하거나 더 나은 성능을 보여줍니다.
Mamba는 Transformer를 대체할 수 있는 범용 시퀀스 백본으로서, 연산 효율, 콘텐츠 기반 추론, 단순 경량화 구조를 동시에 만족시키며 등장했고, 다양한 후속 연구를 통해 확장 및 개선되고 있습니다.
Mamba의 AutoRegressive 방식은 순차 데이터를 처리하는 데 효과적이지만, 컴퓨터 비전 분야에서는 한계점을 가집니다. 왜냐하면 컴퓨터 비전은 전체 수용장(Recep-tive Field)를 활용하기 때문입니다. AutoRegressive는 데이터를 단계별(Step-by-step)으로 처리하므로, 하나의 Forward Pass (순전파)에서 전체적인 문맥(Global Con-text)를 효과적으로 포착하는 데 한계가 있습니다. 컴퓨터 비전 작업에서 Local 영역을 정확하게 예측하기 위해서는 Global 문맥을 고려하는 것이 필수이기 때문에, 기존의 Mamba는 비전 작업에 적합하지 않습니다. AutoRegressive는 시퀀스 데이터를 처리할 때, 현재 시점의 출력이 이전 시점의 출력에만 의존하도로 설계된 것을 말합니다. 이 방식으로 학습할 때, 입력 시퀀스를 한 단계씩 읽어서 그 이전까지의 정보만으로 “다음 토큰”을 예측하도록 손실(loss)을 계산합니다. 그리고 추론할 때 실제 생성 단계에서도 토큰을 하나 예측하면, 그 결과를 다시 모델에 입력하여 다음 토큰을 예측하는 과정을 반복합니다. 이로 인해 한 번의 Forward Pass(순전파)당 한 단계만 처리할 수 있습니다.
Sequence Data는 order(순서)가 중요하지만, 이미지 픽셀은 Sequence 데이터와 동일한 방식으로 순차적 종속성(Sequential Dependency)를 가지지 않습니다. 따라서 기존 Mamba의 순차적 처리 방식은 공간적 데이터를 처리하는 데 비효율적입니다.
### Vision Mamba (Vim)
Vim은 Mamba 구조를 기반으로 양방향 SSM(State Space Model) 공식을 사용했습니다. 토큰을 forward와 backward로 처리하여 더 넓은 Global Context를 포착하고 공간적 이해(Spatial Understanding)을 향상시키고자 했습니다. Vision Mamba (Vim)는 전체 시퀀스를 처리한 후에 예측을 수행해야 해서 실시간 처리가 필요한 응용에서는 Latency(지연)이 크게 증가할 수 있다는 단점이 있습니다. 또한 모델의 구조가 복잡해지면서 학습이 어려워질 가능성도 있고, Overfitting의 위험이 높아질 수도 있습니다.
### VMamba
VMamba는 Cross-Scan Module(CSM)을 도입해서 1D Selective Scan 방식을 활용하는 Mamba 기반 Vision Backbone입니다. CSM은 4방향 Selective Scan을 적용하여 주변 토큰의 정보를 통합하고 더 넓은 Global Context를 포착합니다. VMamba는 Depth-wise Convolution과 Hierarchical Multi-resolution Structure를 포함하는 구조적 변화를 적용했습니다. VMamba는 Cross-Scan Paths에 의해 Receptive Field가 제한되는 문제가 있습니다. 즉, Vmamba의 CSM은 1D인 직선 형태의 Path로만 이루어져있기 때문에, 이 직선이 겹치지 않는 영역에 대한 reception field가 비어 있거나 희박해지기 때문에 완전한 전역 문맥을 한 번에 포착할 수 없습니다.

## 4. 핵심 아이디어
기존 Mamba의 순차 데이터 처리에서의 장점과 Vision Transformer의 Global Context 포착 장점을 결합하기 위해 Mamba 블럭과 Transformer 블럭을 결합했습니다.
Conv Block 내부에 입력을 다시 더해줌으로써 입력과 크게 다른 부분에만 학습 자원을 집중하게 하는 잔차학습 효과와 기울기 소실 완화 효과를 가지게 했습니다.
Stage3와 4에서 Causal Convolution을 Regular Convolution으로 대체함으로써 출력을 계산할 때 상/하/좌/우/대각선 정보들을 동등하게 참고합니다.
Mamba Vision Mixer 블럭 내부에 SSM(Selective State Space Model)이 있는 흐름과 SSM이 없는 흐름으로 나눠서 SSM의 순차적 제약으로 인해 발생하는 정보 손실을 보완했습니다.

## 5. 방법론
### Macro Architecture

### Micro Architecture

### Linear Architecture

## 6. 실험 결과

## 7. 결론

## 8. 느낀점
