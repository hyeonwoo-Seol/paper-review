# SegFormer paper Summary
## 1. 논문 정보
제목: SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers

저자: Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo

## 2. Abstract 요약
SegFormer는 단순하고 효율적이며 강력한 Semantic Segmentation Framework입니다. SegFormer는 Transformer와 경량의 MLP 디코더를 통합한 구조입니다.

SegFormer는 계층적으로 구성된 새로운 Transformer 인코더를 사용하여 Multiscale Feature를 출력합니다. 이 인코더는 Positional Encoding을 사용하지 않기 때문에 Test Resolution과 Training Resolution이 다를 때 발생하는 위치 코드 보간에 대한 성능 저하가 발생하지 않습니다. 그리고 SegFormer는 복잡한 디코더를 사용하지 않고 논문이 제안한 MLP 디코더를 사용합니다. 이 디코더는 서로 다른 층의 정보를 집계하여 Local Attention과 Global Attention을 모두 결합하여 강력한 표현력을 제공합니다.

이 논문에서는 SegFormer-B0 부터 B5까지 모델을 확장했으며, SegFormer-B4는 ADE20K에서 64M 파라미터로 50.3% mIoU를 달성함과 동시에 이전의 최고 방법보다 모델의 크기가 5배 작고 성능이 2.2% 더 우수합니다. SegFormer-B5는 Cityscapes 검증 세트에서 84% mIoU를 기록하고 Cityscapes-C에서 Zero-shot 강건성을 보여주었습니다.

## 3. 문제 정의 및 동기

## 4. 핵심 아이디어

## 5. 방법론

## 6. 실험 결과

## 7. 결론

## 8. 느낀점
