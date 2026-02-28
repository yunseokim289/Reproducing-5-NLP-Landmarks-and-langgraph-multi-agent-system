import torch
import torch.nn as nn
import torch.nn.functional as F # 수학 함수(시그모이드 등)를 위해 추가

# Month 1 과제: Word2Vec Skip-gram 모델 구현
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size # 단어장 크기
        self.embed_dim = embed_dim # 임베딩 차원
        
        # Embedding은 (단어개수 x 차원) 크기의 거대한 숫자 표를 생성
        self.in_embed = nn.Embedding(vocab_size, embed_dim) # in_embed는 중심 단어의 의미를 저장하는 표
        self.out_embed = nn.Embedding(vocab_size, embed_dim) # out_embed는 주변 단어를 구별하기 위한 비교용 표

        initrange = 0.5 / embed_dim # initrange는 하이퍼파라미터(모델의 가중치를 어떤 범위 내에서 무작위로 채울 것인가)
        self.in_embed.weight.data.uniform_(-initrange, initrange) # uniform_함수는 표 안의 값들을 -initrange와 +initrange 사이의 숫자로 골고루 채움
        self.out_embed.weight.data.zero_() # zero_함수는 표 안의 값들을 전부 0으로 채움

    # 순전파 메서드 forward 정의
    def forward(self, input_labels, pos_labels, neg_labels):
       
       """ 
       논문2 section 2.2에서 제안하는 Negative Sampling 이식
       입력 받은 인덱스를 받아서 표에서 벡터(숫자 뭉치)를 꺼냄
       python은 instance field를 함수처럼 호출 가능 
       """

       input_vectors = self.in_embed(input_labels) # 2차원 벡터 (-> bmm호출 위해 3차원 벡터로 변환) , 입력단어wI
       pos_vectors = self.out_embed(pos_labels) # 2차원 벡터 (-> bmm호출 위해 3차원 벡터로 변환) , 정답단어wO(문맥에 맞는 단어)
       neg_vectors = self.out_embed(neg_labels) # 2차원 벡터 (-> bmm호출 위해 3차원 벡터로 변환) , 오답단어wi(문맥에 맞지 않는 단어)
       
       """
       정답(positive) 점수 계산
       bmm함수는 batch matrix multiplication 약어로, 배치(뭉텡이) 단위로 내적(dot product)을 수행
       """
       pos_score = torch.bmm(pos_vectors.unsqueeze(1), input_vectors.unsqueeze(2)).squeeze()

       """
       오답(negative) 점수 계산
       """

       neg_score = torch.bmm(neg_vectors.unsqueeze(1), input_vectors.unsqueeze(2)).squeeze(2)

       return pos_score, neg_score

#Word2Vec Skip-gram 모델이 뱉은 두 점수들을 가지고 모델이 얼마나 틀렸는지 계산하는 손실함수 구현
class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super(NegativeSamplingLoss, self).__init__()

    def forward(self, pos_score, neg_score):   
        pos_loss = F.logsigmoid(pos_score).mean() #정답에 대한 로그 시그모이드 값 계산, 1에 가까울수록 값은 작아짐  
        neg_loss = F.logsigmoid(-neg_score).mean() #오답에 대한 로그 시그모이드 값 계산, 0에 가까울수록 값은 작아짐

        """
        두 값의 합을 최대화하는 것이 목적이나, 
        step4에서 사용할 Optimizer들은 어떤 값을 최소화하도록 설계되어 있기 때문에, 
        수식 전체에 마이너스(-)를 곱함
        즉, 최대화문제를 최소화 문제로 바꿈
        """
        return -(pos_loss + neg_loss) 

    
        
        


"""
테스트 실행 코드
if __name__ == "__main__":
    # 1. 가상의 설정값 (하이퍼파라미터)
    vocab_size = 1000
    emb_dim = 10
    batch_size = 4
    num_neg = 5

    # 2. 모델 및 손실 함수 인스턴스 생성
    model = SkipGramModel(vocab_size, emb_dim)
    criterion = NegativeSamplingLoss()

    # 3. 가상의 입력 데이터 생성 (Random Indices)
    # torch.randint(low, high, size): 범위 내의 정수를 무작위로 생성
    input_labels = torch.randint(0, vocab_size, (batch_size,))
    pos_labels = torch.randint(0, vocab_size, (batch_size,))
    neg_labels = torch.randint(0, vocab_size, (batch_size, num_neg))

    # 4. 모델 실행 (Forward)
    pos_score, neg_score = model(input_labels, pos_labels, neg_labels)

    # 5. 손실 함수 계산
    loss = criterion(pos_score, neg_score)

    # 6. 결과 출력 및 검증
    print("-" * 30)
    print(f"Input Shape: {input_labels.shape}")     # 예상: [4]
    print(f"Positive Score Shape: {pos_score.shape}") # 예상: [4]
    print(f"Negative Score Shape: {neg_score.shape}") # 예상: [4, 5]
    print(f"Final Loss: {loss.item():.4f}")           # 에러 없이 숫자가 나와야 함
    print("-" * 30)
    print("✅ Step 3: 모델 및 손실 함수 연산 테스트 통과!")    

"""
