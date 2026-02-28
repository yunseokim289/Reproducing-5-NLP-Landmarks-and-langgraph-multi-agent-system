import torch
import math
import random
from collections import Counter
#시드값을 42로 고정
random.seed(42) 

class Word2VecDataset:
    # 데이터셋 초기화하는 함수
    def __init__(self, file_path, min_freq = 5):
        # 지정된 경로의 텍스트 파일을 읽음
        with open(file_path,'r',encoding='utf=8') as f:
            # 텍스트를 소문자로 바꾸고 공백 기준으로 단어를 분리
            self.text = f.read().lower().split()
 
        # 등장 횟수가 설정값(5회) 미만인 희귀 단어는 학습에서 제외
        word_counts = Counter(self.text) # Counter는 수백만 개의 단어 리스트를 훑어 {단어번호:나온횟수} 딕셔너리를 생성
        self.vocab = [word for word, count in word_counts.items() if count >= min_freq]

        # 단어를 고유한 숫자 인덱스로 바꾸는 사전 만들기
        self.word2idx = {word: i for i,word in enumerate(self.vocab)}
        # 숫자를 다시 단어로 바꾸기 위한 역방향 사전 만들기
        self.idx2word = {i: word for i,word in enumerate(self.vocab)}
        
        # 텍스트 파일의 글자들을 컴퓨터가 이해하는 숫자(ID)로 변경
        raw_data = [self.word2idx[word] for word in self.text if word in self.word2idx] 
        self.data = raw_data  # 임시 저장
        pairs = self.generate_pairs()  # 중심 단어와 바로 옆에 있는 단어(긍정적 관계)를 짝지어 (중심, 주변) 쌍을 생성

        vocab_indices = list(self.word2idx.values()) # 아무 상관 없는 단어(부정적 관계)를 랜덤하게 골라내기 위해 전체 단어 목록을 준비
        # (중심 단어, 진짜 주변 단어, 가짜 주변 단어)라는 3종 세트를 완성
        self.data = [(c, p, random.choice(vocab_indices)) for c, p in pairs]

    # 데이터셋의 길이를 반환하는 함수
    def __len__(self):
        #논문1 section2 연산 복잡도 식의 T에 해당 
        return len(self.data)
    
    """ 
    논문2 section 2.3 subsampling
    수식 : P(wi) = 1 - sqrt(t/f(wi))를 활용해서 빈도가 높은 단어를 삭제
    """ 
    def _subsampling(self):
        # 논문2 section 2.3에서 제안하는 임계값 t
        t = 1e-5

        # 각 단어의 빈도 계산
        word_counts = Counter(self.data)

        # 전체 토큰 수 
        total_count = len(self.data)

        # subsampling을 거쳐 보존된 단어들만 담을 리스트
        kept_data = []

        for word_idx in self.data:
            # 논문2 section 2.3에서 제안하는 해당 단어의 등장 빈도 f(wi)를 계산
            f_wi = word_counts[word_idx] / total_count

            # 논문2 section 2.3에서 제안하는 단어를 삭제할 확률 P(wi)를 게산
            p_discard  = max(0, 1 - math.sqrt(t / f_wi))

            # p(wi) 보다 난수가 크면 단어를 보존
            if random.random() >= p_discard:
                kept_data.append(word_idx)

        #정제된 데이터 반환        
        return kept_data

    """
    논문1 section 3.2 skip-gram 
    논문2 section 2 sliding window 기법을 사용하여 (center, context) 쌍을 생성

    """        

    def generate_pairs(self, window_size = 5):

        # subsamling을 적용하여 효율적인 학습 데이터 생성
        processed_data = self._subsampling()

        #최종 결과물 리스트
        pairs = []

        # 전체 데이터를 순회하며 중심 단어(center)를 정함
        for i in range(len(processed_data)):
            center_word = processed_data[i]

            # window(단어 간의 최대 거리)의 시작과 끝 범위를 결정, 중심단어 기준으로 왼쪽으로 window_size만큼 오른쪽으로 window_size만큼 살펴보도록 설정

            start = max(0, i - window_size)
            end = min(len(processed_data), i + window_size + 1)

            # 범위 내의 주변 단어(context)를 순회
            for j in range(start, end):
                if i != j:
                    context_word = processed_data[j]

                    pairs.append((center_word, context_word)) 
        #최종 생성된 (center,context)쌍 리스트 반환
        return pairs

    # 특정 인덱스의 데이터를 꺼내주는 메서드
    def __getitem__(self, idx):
        return self.data[idx]


    


        


            
"""
테스트 실행 코드
if __name__ == "__main__":
    # 1. 데이터셋 객체 생성 (Step 1 결과물 활용)
    dataset = Word2VecDataset("data/toy_data.txt") 
    
    # 2. Step 2 실행: 윈도우 사이즈 2로 설정
    train_pairs = dataset.generate_pairs(window_size=2)
    
    # 3. 결과 분석 [cite: 21]
    print(f"Original Length: {len(dataset.data)}")
    print(f"Subsampled Length: {len(dataset._subsampling())}") # 줄어들었는지 확인
    print(f"Total Training Pairs: {len(train_pairs)}")
    print(f"Sample (Input, Output): {train_pairs[:3]}") # 튜플 형태 확인

"""

        

