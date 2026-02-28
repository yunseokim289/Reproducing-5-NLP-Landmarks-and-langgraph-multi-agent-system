import torch
import torch.nn as nn # 파이토치의 신경망 도구들을 불러옴

# ELMo 논문의 핵심인 BiLM의 기본 구조
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__() 
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim) # 단어를 고차원 벡터로 바꿈 
        # batch_first = True는 데이터가 (묶음, 길이, 특징) 순서로 들어온다는 설정 
        # 서로 간섭하지 않는 두 개의 LSTM을 만듦.
        self.f_lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first = True) # L_layer LSTM(정방향용)
        self.b_lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first = True) # L_layer LSTM(역방향용)

        self.fc = nn.Linear(hidden_dim, vocab_size) # LSTM이 뽑아낸 특징을 다시 '단어 개수'만큼의 숫자로 펼쳐주는 마지막 층
    
    # forward는 데이터가 모델을 통과하는 실제 순서를 정의
    # 기존 gen_rnn.py가 쓰는 함수 (정방향만 사용)
    def forward(self, x, hidden):
        out = self.embedding(x) # 단어를 고차원 벡터로 바꿈

        out, hidden = self.f_lstm(out, hidden) # f_LSTM 통과(out에는 결과물, hidden은 다음 시점으로 전달할 기억)

        out = self.fc(out) # 마지막 층을 통과하여 각 단어별 점수(Logits) 계산
        return out, hidden
    
    # train_rnn.py가 쓰는 함수 (정방향, 역방향 모두 사용. 즉 양방향 사용)
    def forward_bi(self, x_f, x_b, hidden_f, hidden_b):
        # Forward Pass(정방향) 사용
        emb_f = self.embedding(x_f)
        out_f, hidden_f = self.f_lstm(emb_f, hidden_f)

        """
        [logits_f의 의미]
        논문 Section 3.1 : likelihood p(tk | t1, ..., tk-1) 계산
        과거의 문맥을 보고 현재 단어를 맞춤 (History -> Target)
        """
        logits_f = self.fc(out_f)

        # Backward Pass(역방향) 사용
        emb_b = self.embedding(x_b)
        out_b, hidden_b = self.f_lstm(emb_b, hidden_b)
        
        """
        [logits_b의 의미]
        논문 Section 3.1 : likelihood p(tk | tk+1...tN) 계산
        미래의 문맥을 보고 현재 단어를 맞춤 (Future -> Target)
        """
        logits_b = self.fc(out_b)

        return logits_f, logits_b, hidden_f, hidden_b

    def init_hidden(self, batch_size, device):
        # LSTM은 처음 시작할 때 아무 기억이 없는 상태(0)에서 시작해야 됨
        # (h0, c0) 즉, hidden state(단기 기억)와 cell state(장기 기억)를 0으로 채워진 텐서로 만듦
        # .to(device)는 생성한 텐서를 device에서 계산(GPU 쓸때 필수, 디폴트 : CPU)

        # 정방향용 기억(h_f)
        h_f = (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device), torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))
        # 역방향용 기억(h_b)
        h_b = (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device), torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))

        return h_f, h_b

                          
