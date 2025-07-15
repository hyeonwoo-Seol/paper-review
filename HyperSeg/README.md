# HyperSeg Summary
## 논문 정보
제목: HyperSeg: Hybrid Segmentation Assistant with Fine-grained Visual Perceiver

저자: Cong Wei, Yujie Zhong, Haoxian Tan

## Abstract 요약
복잡한 추론 분할에 대해서 이미지와 비디오 모두 적응하기 어려운 한계점으로 인해, fine-grained 수준의 vision-language 상관관계를 정확히 이해하는 데 어려움이 있습니다.

이에 이 논문은 HyperSeg를 제안합니다. HyperSeg는 VLLM 기반 pixel-level 이미지와 비디오 인식을 위한 범용 분할 모델입니다.

일반적인 분할 작업을 포함하고, 강력한 추론 능력과 world knowledge가 필요한 복잡한 추론 인식 작업을 포함합니다.

VLLM의 인식 능력과 세밀한 시각 정보를 전부 활용하기 위해, HyperSeg는 다양한 분할 작업을 위해 hybrid entity Recognition과 세밀한 시각 인식 모듈을 통합합니다.

Temporal Adapter와 결합함으로써, HyperSeg는 temporal 정보의 포괄적인 이해가 가능합니다.
## 문제 정의 및 동기
기본적인 Vision-Language 정렬 방법의 한계로 인해 세부적인 정보 이해가 어렵습니다.

VLLM는 이미지와 비디오 도메인에서 모두 가능한 범용 분할 프레임워크로써 활용되기 어렵고, 복잡한 비디오 추론 능력이 부족합니다.



## 핵심 아이디어
### hybrid entity recognition
Generation과 Decoding 단계에서 LLM을 활용하는 방식입니다.

VLLM이 예측된 객체 이름 뒤에 Mask Token을 생성하도록 지시합니다.

그림3-c

VLLM은 시각 입력에 존재하는 모든 객체를 먼저 생성한 뒤, Semanticly Enhanced Mask Token을 생성합니다. 이 토큰은 이미지에 대한 통합된 의미 정보를 포함하고, 분할 예측기의 입력으로 사용되어 Segmentation Mask를 생성합니다.

### fine-grained visual perceiver modules
그림4

Multi-Scale Visual 특징을 세분화된 토큰으로 융합합니다.

이를 통해 과도한 계산 비용 없이 사전 학습된 VLLM에 풍부한 fine-grained visual 정보를 주입할 수 있습니다.

피라미드 Visoin Encoder F_seg를 사용해서, 시각 입력 V로부터 이미지 특징 f_img를 얻습니다. f_img 는 세부 정보에 민감한 특징이 있습니다.

FVP 모듈은 Conditional Weighted Cross-Attention을 통해 각 토큰을 Eq5~6처럼 풍부화합니다.

Eq5, Eq6

위 수식에서 MHCA는 Multi-Head Cross-Attention Layer 이고, G_p는 projection Function이고, tanh는 Normalizaton Function 입니다.

tanh(MLP()는 조건부 가중치로서 사용되고, 이전 토큰 P_j-1에 잔차 연결을 수행하기 전에 이 조건부 가중치를 풍부화된 정밀 시각 토큰 P\hat_j에 곱합니다.

이 논문은 다양한 Multi-Scale Image Feature에 대한 적응과 학습 안정성을 유지하기 위해 초기 가중치 값을 0으로 설정했습니다.

### temporal adapter
복잡한 비디오 인식 과제를 해결하기 위해, 시간 축에서 global prompt Aggregation과 Local Space-time Information Injection을 활용하고자 합니다.

Global Prompt Aggregation. 현재 프롬프트 임베딩 E_p에 시간 축을 따라 적응적 평균 풀링 전략을 사용해서, 이전 T 프레임들의 전역 객체 정보와 시간 정보를 집약합니다.

Eq7

Local Space-Time Information Injection. 
## 방법론
### overall architecture
information (Sec 3.3). The VLLM takes three types of inputs: visual tokens encoded by the CLIP encoder, renewed
fine-grained tokens, and prompt tokens for diverse instructions. The output embeddings of semantically enhanced
mask tokens (Sec 3.2) and prompt tokens are further fed into
the segmentation predictor for final segmentation results.
Besides, we utilize the space-time information propagation
and global prompt aggregation for comprehensive video understanding (Sec 3.4). We train the LLM with LoRA for
efficient parameter tuning.

### 

## 실험 결과

## 결론

## 느낀점

