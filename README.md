# Learning Multimodal Empathic Responses from the IEMOCAP Dataset

# 프로젝트 소개 
기존 감정인식 멀티모달을 넘어 상대방의 3가지 모달리티(표정, 어조, 발화)에 대해 반응을 학습하는 모델을 제안한다. 또한, 공감이라는 주관적인 평가에 의존하는 것이 아닌 accuracy로도 함께 평가할 수 있도록 얼굴 랜드마크 값을 사용한다. 

이 프로젝트는 IEMOCAP 데이터셋의 남녀 대화 쌍을 이용하여 '여성의 발화(입력) → 남성의 반응(출력)'을 예측한다. 입력과 출력을 모두 (텍스트, 음성, 영상, 얼굴 랜드마크) 특징 벡터로 구성하여, 모달리티에 대한 공감 반응 모달리티를 학습한다. 이는 기존의 분류 문제를 넘어, 벡터 공간에서 반응의 유사성을 측정하는 회귀(Regression) 문제로 접근하여 보다 객관적이고 정량적인 평가가 가능하다는 점을 보여준다.

# 프로젝트 특징
반응 예측 모델: 단일 발화의 감정 분류를 넘어, 대화 상대방의 다음 반응을 멀티모달 신호로 직접 예측힌다.
객관적 평가 지표: 주관적인 공감 평가를 보완하기 위해, 예측된 얼굴 랜드마크와 실제 랜드마크 간의 정확도(Accuracy)를 평가지표로 사용한다.
멀티모달 통합: 텍스트의 의미, 음성의 운율, 얼굴의 표정 정보를 모두 융합하여 상호작용을 모델링한다.
회귀 기반 학습: 반응 벡터 자체를 예측하는 회귀(Regression) 방식으로 학습하여, 정량적인 오차(MSE) 측정이 가능하다.

🛠️ 모델 구조 및 학습 파이프라인
데이터 준비: IEMOCAP 데이터셋에서 '여성 발화 → 남성 반응'으로 이어지는 대화 쌍(ID)을 추출한다.
특징 벡터 로딩: 각 발화 ID에 해당하는 텍스트(Text), 음성(Voice), 영상(Video), 그리고 얼굴 랜드마크(Landmark) 특징 벡터를 로드한다.
모델 학습: 여성의 통합 특징 벡터를 입력받아, 남성의 반응 특징 벡터(특히 랜드마크 포함)를 예측하도록 모델을 학습한다.
손실 계산: 예측된 벡터와 실제 벡터 간의 오차를 MSE(Mean Squared Error) 등의 손실 함수로 계산하여 모델을 최적화한다.

+---------------------------------+      +--------------------------------+      +---------------------------------+
|      [입력] 여성 발화           |      |      [모델] 멀티모달 융합 인코더  |      |      [출력] 남성 반응 예측       |
| (Text + Audio + Video 특징 벡터) |  ->  | (Multimodal Fusion Encoder)    |  ->  | (Text + Audio + Video 특징 벡터) |
+---------------------------------+      +--------------------------------+      +---------------------------------+
                                                     |
                                                     | Loss Calculation (MSE / Cosine Similarity)
                                                     |
                                     +---------------------------------+
                                     |      [정답] 실제 남성 반응       |
                                     | (Text + Audio + Video 특징 벡터) |
                                     +---------------------------------+
데이터 준비: IEMOCAP 어노테이션을 파싱하여 '여성 발화 → 남성 반응'으로 직접 이어지는 모든 대화 쌍의 ID를 추출합니다.
특징 벡터 로딩: 각 발화 ID에 해당하는 텍스트, 음성, 시각 특징 벡터를 로드합니다. (GraphSmile 등에서 제공하는 사전 추출된 벡터 활용)
모델 학습: 여성의 특징 벡터를 입력받아 남성의 반응 특징 벡터를 예측하도록 모델을 학습시킵니다.
평가: 테스트 데이터셋에서 예측된 벡터와 실제 벡터 간의 유사도를 측정하여 모델의 성능을 평가합니다.
⚙️ 환경 설정 (Installation)
리포지토리 복제:

Bash

git clone https://github.com/yujin0410/EmpathicAI.git
cd EmpathicAI
필요 라이브러리 설치:

Bash

pip install -r requirements.txt

Python 3.8
# How to Run
1. Data Preparation
IEMOCAP 데이터셋에서 대화 쌍을 생성한다. 

Bash

python scripts/create_dialogue_pairs.py --data_path path/to/IEMOCAP --output_path ./data
2. 모델 학습 (Training)
준비된 데이터를 사용하여 모델 학습을 시작합니다.

Bash

python run.py \
    --data_path ./data/dialogue_pairs.pkl \
    --feature_path path/to/features \
    --model_name EmpathicAI_Predictor \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --loss_fn MSE
