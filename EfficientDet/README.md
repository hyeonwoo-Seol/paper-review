# EfficientDet paper Summary
## 1. 논문 정보
- 제목: EfficientDet: Scalable and Efficient Object Detection
- 저자: Mingxing Tan, Ruoming Pang, Quoc V. Le

## 2. Abstract 요약
- 객체 탐지를 위해 EfficientNet을 기반으로 EfficientDet을 개발했고, 다음의 최적화 기법을 사용했습니다. 우선, weighted bi-directional feature pyramid network (BiFPN)을 사용해서 빠르고 쉬운 다중 스케일 특징 융합을 가능하게 합니다. resolution, depth, width를 Backbone, Feature Network, Box/Class Prediction Network 전체에 균일하게 확장하는 Compound Scaling 방법을 사용합니다. 이를 통해 다양한 자원 제약 환경에서도 기존의 기법보다 우수한 효율성을 달성했고, 단일 모델 및 스케일 설정에서 EfficientDet-D7은 COCO test-dev 에서 State-of-the-Art를 달성했습니다.

## 3. 문제 정의 및 동기
- 기존의 State-of-the-Art Detection 모델들은 자원을 많이 사용하면서 높은 정확도를 달성해, 로봇공학이나 낮은 레이턴시를 요구하는 환경에서 사용하기에 어렵습니다. one-stage, anchor-free-detectors, compress model 기법들은 더 나은 효율성을 달성하지만 동시에 정확도를 낮춥니다. 또한 대부분의 연구들은 특정되고 제한된 범위의 자원 요구사항에만 초점을 맞추고 있어서, 실제 환경인 모바일 장치부터 데이터센터까지의 다양한 자원 제약을 고려해야 합니다.

## 4. 핵심 아이디어
- 기존의 연구들은 서로 다른 스케일의 특징을 융합하기 위해, 서로 다른 해상도의 입력 특징들을 단순히 더했습니다. 이는 각 특징이 융합된 출력에 기여하는 비중이 동등하지 않다는 것을 고려하지 않았기 때문에, 이 논문은 weighted bi-directional feature pyramid network (BiFPN)을 제안합니다. BiFPN은 학습 가능한 가중치를 도입하여 각 입력 특징의 중요도를 학습하고, top-down과 botton-up 방식의 다중 스케일 특징 융합을 반복적으로 수행합니다.
- Object detection을 위한 compound scaling 기법을 통해, resolution, depth, width를 backbone, feature network, box/class prediction network 전체에 걸쳐서 확장합니다.
- one-stage detector design은 two-stage detector design 보다 정확도는 낮지만 단순하고 효율적입니다. 이 논문은 주로 one-stage detector design을 사용하면서 효율성과 높은 정확도를 달성함을 보여줍니다.

## 5. 방법론

## 6. 실험 결과

## 7. 결론

## 8. 느낀점
