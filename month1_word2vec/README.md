# LILAB Intern Project

## Week 1: Environment Setup & Sanity Check
- Environment: Python 3.10, PyTorch 2.5.1 installed.
- Repository Structure: Created `src`, `configs`, `scripts`, `data`, `runs` folders.
- Sanity Check: Successfully ran `train.py` with Mixed Precision (AMP) enabled.
- Result: `Sanity Check 완료! Loss: -456.4113` (Run on CPU).

## 2주 차 진행 상황
- Step 1: 데이터 전처리 및 단어장 구축 완료
- 원문 데이터(`toy_data.txt`) 로드 및 소문자 정규화 수행 
- Hyperparameter: `min_freq=5` 적용하여 희귀 단어 제거
- Result: 단어장 크기(V) 3324개, 전체 학습 데이터(T) 약 2243862개 확보

## 3주 차 진행 상황 : 데이터셋 구축 및 서브샘플링 (Subsampling)

### 1. 고빈도 단어 서브샘플링 (Subsampling)
- 구현 내용: 
  - 학습 효율을 저해하는 고빈도 단어(예: 'the', 'a' 등)를 확률적으로 제거하기 위해 삭제 확률 
    P(w_i) = 1 - \sqrt{t/f(w_i)} 수식을 적용 
  - 논문에서 권장하는 임계값(Threshold)인 t = 1e-5를 하이퍼파라미터로 설정하였습니다. 
- 기대 효과: 전체 토큰 수를 최적화하여 학습 속도를 높이고, 희소 단어(Rare words)에 대한 벡터 표현 품질을 개선

### 2. 학습용 데이터 쌍 생성 (Skip-gram)
- 기법: 슬라이딩 윈도우 (Sliding Window) 적용 
- 구현 상세: 
  - 윈도우 크기를 5로 설정하여 중심 단어와 주변 단어의 관계를 (Input, Output) 형태의 인덱스 쌍으로 변환
  - 재현성: 실험 결과의 일관성을 위해 `random.seed(42)`를 사용하여 샘플링 결과를 고정하였습니다. 
- 최종 검증(Sanity Check) 결과:
  - Original Length: 2,243,862개
  - Subsampled Length:  304281개
  - Total Training Pairs: 1219070개
  - Sample (Input, Output): [(5, 6), (5, 13), (6, 5)]

## 4주 차 진행 상황 : Skip-gram 모델 구현 및 Negative Sampling 이식

### 1. Skip-gram 아키텍처 설계
- 구현 내용: 
  - 논문의 skip-gram 구조를 정밀하게 이식하여 중심 단어(`in_embed`)와 주변 단어(`out_embed`) 테이블 분리 구현
  - 가중치 초기화: `in_embed`는 Uniform Distribution, `out_embed`는 Zero Initialization을 적용하여 논문 표준 준수
- 기대 효과: 단어 간의 유사도를 벡터 공간상의 거리로 학습할 수 있는 신경망 기반을 구축하고 초기 학습 안정성 확보

### 2. Negative Sampling 및 커스텀 손실 함수
- 기법: Negative Sampling (NEG) 직접 구현
- 구현 상세: 
  - `torch.bmm` 및 `unsqueeze` 연산을 활용하여 배치(Batch) 단위의 효율적인 병렬 내적 연산 로직 구현
  - `F.logsigmoid`를 적용하여 수식의 로그-시그모이드 연산에 대한 수치적 안정성 확보
- 최종 검증(Sanity Check) 결과:
  - 실험 환경: Batch Size 4, Embedding Dim 10, Negative Samples 5
  - Shape Check: Positive Score `torch.Size([4])`, Negative Score `torch.Size([4, 5])` 일치 확인
  - Result: `✅ Step 3: 모델 및 손실 함수 연산 테스트 통과!` (Final Loss 약 4.1589 산출)

## 5주 차 진행 상황 : 대규모 모델 학습 및 최적화 

### 1. GPU 가속 및 연산 최적화
- 실험 환경: NVIDIA GeForce RTX 4090 GPU 서버 활용
- 구현 상세:
  - AMP(Automatic Mixed Precision) 적용: FP16 연산을 수행하여 학습 속도 및 메모리 효율 최적화
  - 데이터 파이프라인: (Center, Pos, Neg) 3종 세트 구성

### 2. 학습 수행 및 결과
- Hyperparameter: `Batch Size=4096`, `Learning Rate=0.001`, `Epochs=5`, `Embedding Dim=100`
- Result:
  - Final Loss: 0.3194 (최종 에폭 완료 및 `word2vec_model.pth` 저장 완료)

## 6주 차 진행 상황 : Harry Potter Word2Vec Project

본 프로젝트는 해리포터 영화 시리즈 텍스트 데이터를 바탕으로 단어 임베딩 모델(Word2Vec)을 구축하고, 모델이 단어 사이의 논리적/문맥적 관계를 얼마나 잘 이해하는지 평가합니다.

## 평가 결과 (Evaluation)
### 1. 유사도 테스트 (Similarity)
- `emma` 입력 시 출력 시 emma와 가장 유사한 단어(top-5) 출력

### 2. 유추 테스트 (Analogy)
- 질문: harry : potter = daniel : ?
- 결과: radcliffe
- 분석: 이름-성 관계 도출

## 7주차 진행 상황 : Word2Vec Embedding Visualization

본 문서는 학습된 Word2Vec 모델의 임베딩 벡터를 시각적으로 분석한 결과를 기록합니다.

## 1. 시각화 개요
- 목적: 100차원의 단어 벡터를 2차원 평면으로 투영하여, 단어 간의 의미적 거리와 군집(Cluster) 형성 여부를 육안으로 검증함.
- 방법: PCA (Principal Component Analysis) 기법을 사용하여 차원 축소.
- 대상 단어: `harry`, `ron`, `hermione` 등 주요 등장인물 및 `movie`, `director` 등 일반 명사.

## 2. 분석 결과 (Key Findings)
- 군집화(Clustering):
  -"harry","potter" and "emma", "watson"처럼 비슷한 단어들끼리 군집화를 이룸
  -사람 이름들(harry, daniel, emma 등)은 한쪽에 뭉쳐 있고, movie, director, magic,  --hogwarts 같은 일반 명사들은 그들과 조금 떨어진 다른 쪽에 뭉쳐 있음.
  이는 모델이 사람과 사물/개념의 의미적 차이를 구분하고 있다는 증거

결과 : src/word_embedding_plot.png




  
  


