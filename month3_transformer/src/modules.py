import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

# [Section 3.2.2] Multi-Head Attention 구현
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super().__init__() # 파이썬의 상속 문법. nn.Module(부모)가 가진 기능을 그대로 물려받기 위해 반드시 초기화해야됨.
        
        # assert 조건, 메시지 : d_model이 n_head로 나누어 떨어지지 않으면 시작 전에 강제로 에러를 내서 멈춤
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_head = d_model // n_head # 각 Head가 담당할 차원의 크기
        self.n_head = n_head # Head의 개수
        self.d_model = d_model # 입력 벡터의 차원

        # [Section 3.2.2] Linear projections
        # 입력 벡터(d_model)을 Q, K, V 공간으로 투영하는 선형층
        # nn.Linear는 내부적으로 Weight와 Bias을 학습
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # 각 Head의 결과를 합친 후 마지막으로 통과시키는 선형층
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask = None):
        batch_size= q.size(0) # 텐서의 크기를 반환

        # 입력을 n_head개로 쪼개서 병렬 처리하기 위한 차원 변환 과정
        # seq_len을 -1로 설정해서 길이 자동으로 맞춤
        # view는 텐서의 모양 변경, transpose는 행렬 곱을 위해 1번 차원과 2번 차원의 위치를 바꿈
        Q = self.w_q(q).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2) # QW^Q
        K = self.w_k(k).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2) # KW^K
        V = self.w_v(v).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2) # VW^V

        scores = torch.matmul(Q, K.transpose(-1,-2)) # [Section 3.2.2] QW^Q와 KW^K의 단어들끼리 얼마나 친한지 내적(dot product)
        
        scores = scores / math.sqrt(self.d_head) # 내적값이 너무 커지는 것을 방지하기 위해 Scaling

        """
        Encoder (독해 담당): 입력된 문장을 읽고 "이게 무슨 뜻이지?" 하고 문맥을 파악해서 압축하는 녀석
        Decoder (작문 담당): Encoder가 이해한 내용을 바탕으로 "그럼 다음엔 무슨 말이 와야 하지?" 하고 결과를 생성하는 녀석
        """

        # [section 3.2.3] Causal Masking : Decoder 구조에서는 미래의 단어를 미리 보면 안되므로 가려줘야됨.

        if mask is not None:
            # 내적의 결과 텐서에 mask(봐도 되는건 1, 그렇지 않은 건 0으로 채워진 행렬)를 덮어서 mask가 0인 부분을 -1e0로 채움
            # 그러면 나중에 Softmax를 통과하면 0에 수렴에서 원천 차단
            scores = scores.masked_fill(mask == 0, -1e9) 
 
        # 마지막 차원을 기준으로 값들을 softmax함수에 넣어서 0~1 사이의 확률값으로 변환
        attn_weights = F.softmax(scores, dim = -1)

        out = torch.matmul(attn_weights, V) # softmax 결과에 V을 곱한게 최종 Attention(QW^Q, KW^K, VW^V)

        out = out.transpose(1,2).contiguous().view(batch_size, -1, self.d_model) # Concat(head1, ...)
        
        return self.w_o(out), attn_weights # Multi-Head Attention : MultiHead(Q, K, V) == Concat(head1, ...) W^O와 attn_weights를 리턴


# [Section 3.1] Encoder Layer 구현
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout = 0.1):
        super().__init__()

        # Encoder는 자기 자신만 보므로 Attention 1개
        # 부품 1 : Multi-Head Attention
        self.self_attn = MultiHeadAttention(d_model, n_head)

        # 부품 2 : FFN
        self.ffn = PostionWiseFeedForward(d_model, d_ff, dropout)
        
        # 부품 3 : Layer Normalization 1
        # 층 정규화 : 각 샘플의 평균과 분산을 맞춰줘서 학습을 안정시킴
        self.norm1 = nn.LayerNorm(d_model)

        # 부품 4 : Layer Normalization 2
        self.norm2 = nn.LayerNorm(d_model)

        # 부품 5 : Dropout
        self.dropout = nn.Dropout(dropout)
    
    # EncoderLayer의 mask (Source Mask) : 
    # 문장 길이가 짧아서 뒤에 채워 넣은 의미 없는 빈칸(Padding) 보지 마!
    def forward(self, x, mask): 
        # 1. self-attention (source mask 사용)
        # q, k, v 모두 x (자기 자신)
        # 식 : LayerNorm(x + Sublayer(x)) 두 번 사용, 처음 x는 임베딩 벡터이고 두번째 x는 정규화된 결과값
        # 객체 대입한 instance field()시 forward()호출됨 
        attn_out, _ = self.self_attn(x, x, x, mask) # Sublayer(x) 리턴
        
        # Residual Connection (잔차 연결)
        # Multi-Head Attention의 결과값은(attn_out) 원래 정보(x)와 잔차 연결(더해짐)됨
        # 그 후 층 정규화시킴
        x = self.norm1(x + self.dropout(attn_out)) # LayerNorm(x + Sublayer(x)) # Add & Norm

        # 2. Feed Forward
        # [FFN Sub-layer] 식 : LayerNorm(x + Sublayer(x)) 구현
        ffn_out = self.ffn(x) # Sublayer(x) 리턴

        # Residual Connection (잔차 연결)
        # FFN의 결과값은(ffn_out) 원래 정보(x)와 잔차 연결(더해짐)됨
        # 그 후 층 정규화시킴
        x = self.norm2(x + self.dropout(ffn_out)) # LayerNorm(x + Sublayer(x)) # Add & Norm

        return x




# [Section 3.3] Position-wise Feed-Forward Networks 구현
# 식: FFN(x) = max(0, xW1 + b1)W2 + b2

class PostionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super().__init__()

        # 첫 번째 선형층 (확장)
        # d_model(256) -> d_ff(1024, 보통 4배)
        # 입력을 더 높은 차원으로 뻥튀기해서 특징을 풍부하게 만듦
        self.w_1 = nn.Linear(d_model, d_ff)
        
        # 두 번째 선형층 (압축)
        # d_ff(1024) -> d_model(256)
        # 다시 원래 크기로 줄여서 다음 층에 전달하기 좋게 만듦
        self.w_2 = nn.Linear(d_ff, d_model)
        
        # Dropout : 학습 시 랜덤하게 뉴런을 꺼서 과적합(Overfitting)을 방지
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x)))) # FFN(x) = max(0, xW1 + b1)W2 + b2 구현 후 리턴


# [Section 3.5] Positional Encoding 구현
# 식1 : PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
# 식2 : PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model) # (max_len, d_model) 크기의 0으로 가득 찬 행렬 생성

        # 0부터 max_len-1까지의 숫자를 생성(위치 인덱스: pos)
        # unsqueeze(1)은 차원 늘려줌
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        
        # 1 / 10000^(2i / d_model) == exp(-2i · ln(10000) / d_model ) 을 구현 
        # log space에서 계산하는 것이 수치적으로 더 안정적이기 때문
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 행렬[행, 열]
        # 행렬 pe에서 모든 행 가져오고, 짝수 인덱스 열만 가져오라.
        pe[:, 0::2] = torch.sin(position * div_term)
        # 행렬 pe에서 모든 행 가져오고, 홀수 인덱스 열만 가져오라.
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        
        # register_buffer는 nn.Module에서 받은 instance method
        # self.pe란 instance field 생성 후 pe(텐서)를 저장
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 텐서에서의 +는 x와 self.pe의 길이가 맞아야됨.
        return x + self.pe[:, :x.size(1)] # :x.size(1)은 x의 문장의 길이


# [Session 3.1] Figure 1의 Encoder Layer 구현

class DecoderLayer(nn.Module):

    def __init__(self, d_model, n_head, d_ff, dropout = 0.1):
        super().__init__()
       
        # 1. Masked Self-Attention 
        self.self_attn = MultiHeadAttention(d_model, n_head)
        
        # 2. Self- Attention (Encoder 결과를 쳐다보는 Cross-Attention)
        self.cross_attn = MultiHeadAttention(d_model, n_head)

        # 3. FFN
        self.ffn = PostionWiseFeedForward(d_model, d_ff, dropout)
        
        # 4. 층 정규화 3개 필요
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    
        # 5. Dropout
        self.dropout = nn.Dropout(dropout)
    
    # 인코더 출력(enc_output)과 원본의 빈칸(padding)보지마(src_mask), 미래 보지마(tgt_mask) 
    # 이 3개를 추가로 받음        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        
        # 1. Masked Self-Attention (미래 정보 가리기 : tgt_mask)
        attn_out, _ = self.self_attn(x, x, x, tgt_mask) #  Sublayer(x)
                
        # Add & Norm
        x = self.norm1(x + self.dropout(attn_out)) # LayerNorm(x + Sublayer(x))

        # 2. ★인코더와의 차이점 : Cross-Attention (인코더 훔쳐보기 +
        # 원본의 빈칸 가리기 : src_mask)
        cross_out, _ = self.cross_attn(x, enc_output, enc_output, src_mask) #  Sublayer(x)

        # Add & Norm
        x = self.norm2(x + self.dropout(cross_out)) # LayerNorm(x + Sublayer(x))

        # 3. Feed Forward
        ffn_out = self.ffn(x) #  Sublayer(x)

        # Add & Norm
        x = self.norm3(x + self.dropout(ffn_out)) # LayerNorm(x + Sublayer(x))

        return x

# [Section 3] The Transformer - model architecture

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_ff, max_len, n_layer, dropout = 0.1):
        super().__init__()

        # 1. 임베딩 층 (소스 언어, 번역할 원본 문장용 & 타겟 언어,번역된 결과 문장용 따로) -> 언어가 다르기 때문
        # 임베딩 층은 단어 -> 벡터(임베딩) 해주는 단어장(사전)
        # vocab_size : 총 단어 개수
        # d_model : 입력 벡터의 차원

        self.src_embedding = nn.Embedding(vocab_size, d_model) # 단어를 고차원 벡터로 바꿈(임베딩 층)

        self.tgt_embedding = nn.Embedding(vocab_size, d_model) # 단어를 고차원 벡터로 바꿈(임베딩 층) 
        
        # 2. 위치 인코딩
        self.pos_enc = PositionalEncoding(d_model, max_len) # 위치 정보 생성
        
        # 3. dropout
        self.dropout = nn.Dropout(dropout)
        

        # 4. ★Encoder stack (인코더 층 쌓기)
        # 파이썬 리스트([])대신 nn.ModuleList를 써야 pytorch가 관리해줌
        # n_layer만큼 만복해서 EncoderLayer 생성해서 리스트 안에 담음
        self.encoders = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(n_layer)
        ])
        
        # 5. ★Decoder stack (디코더 층 쌓기)
        # 파이썬 리스트([])대신 nn.ModuleList를 써야 pytorch가 관리해줌
        # n_layer만큼 만복해서 DecoderLayer 생성해서 리스트 안에 담음
        self.decoders = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(n_layer)
        ])

        # 6. 출력층 (벡터 -> 단어 확률)
        # d_model 크기의 벡터를 다시 vocab_size(단어 개수)로 넓혀서 
        # 어떤 단어일 확률이 높은지 점수(Logits)를 매김
        self.final_linear = nn.Linear(d_model, vocab_size)
    
    # [Mask 생성 함수 1] 소스 문장의 빈칸(pad) 가리기
    def mask_src_mask(self, src):
        
        # src는 소스문장이고 빈칸은 0으로 채워진 리스트
        # 패딩은 False, 나머지는 True로 채워진 리스트를 리턴
        return (src != 0).unsqueeze(1).unsqueeze(2)


    # [Mask 생성 함수 2] 타겟 문장의 미래 가리기 & 빈칸(문장 끝난 뒤의 0) 가리기
    def mask_tgt_mask(self, tgt):
        
        # 1. 빈칸(padding) 마스크
        # tgt는 타겟문장이고 빈칸은 0으로 채워진 리스트
        # 패딩은 False, 나머지는 True로 채워진 리스트를 리턴
        pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)   
        
        # 2. 미래 가리기
        seq_len = tgt.size(1) # tgt의 문장 길이 리턴
        
        # torch.ones()는 가로세로 길이가 seq_len인 정사각형 행렬을 만들고, 숫자 1로 가득 채움
        # torch.tril() : 대각항 기준 위쪽 상 삼각형'U'을 0으로 채움(과거'L'는 보고 미래'U'는 가림)
        # 대각항은 현재(지금 모습)
        # type_as(tgt): tgt가 GPU에 있으면 마스크도 GPU로 옮기고, 타입도 똑같이 맞춤
        tril = torch.tril(torch.ones(seq_len, seq_len)).type_as(tgt)

        return pad_mask & tril # &은 파이토치의 and 논리연산자
        



    def forward(self, src, tgt):

        # 1. 마스크 준비
        src_mask = self.mask_src_mask(src)
        tgt_mask = self.mask_tgt_mask(tgt)

        # 2. 소스 언어용 embedding 실행 
        # 입력 -> 임베딩 -> 위치정보 -> 드롭아웃
        enc_out = self.dropout(self.pos_enc(self.src_embedding(src)))
        
        # 3. Encoder layer 통과
        for layer in self.encoders:
            enc_out = layer(enc_out, src_mask)

        # 4. 타겟 언어용 embedding 실행 
        # 입력 -> 임베딩 -> 위치정보 -> 드롭아웃
        dec_out = self.dropout(self.pos_enc(self.tgt_embedding(tgt)))   

        # 5. Decoder Layer 통과 (★Cross-Attetion 발생)
        for layer in self.decoders:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)
            
        # 6. 결과 예측 (Logits 출력)
        logits = self.final_linear(dec_out)

        return logits 



"""
# 테스트 실행 코드
if __name__ == "__main__":
    # Transformer 조립 테스트
    print("\n[Test] Transformer Model Assembly Check...")
    
    # 1. 설정값 (작게 잡아서 테스트)
    vocab_size = 100
    d_model = 32
    n_head = 4
    d_ff = 64
    max_len = 20
    n_layer = 2
    
    # 모델 생성
    model = Transformer(vocab_size, d_model, n_head, d_ff, max_len, n_layer)
    
    # 2. 가짜 데이터 (Batch=2)
    # src(영어): 길이 10
    src = torch.randint(1, vocab_size, (2, 10))
    # tgt(한글): 길이 12 (길이가 달라도 돌아가는지 확인!)
    tgt = torch.randint(1, vocab_size, (2, 12))
    
    # 3. 실행
    logits = model(src, tgt)
    
    print("Input Src:", src.shape)      # (2, 10)
    print("Input Tgt:", tgt.shape)      # (2, 12)
    print("Output Logits:", logits.shape) # (2, 12, 100)
    
    # 검증
    # 출력 크기는 (Batch, Tgt_Len, Vocab_Size)여야 함
    assert logits.shape == (2, 12, 100), "Shape Mismatch!"
    print("✅ PASS: Transformer 모델 완성! 서버 열리면 바로 학습 가능!")
        
"""








