import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from src.month2.dataset_rnn import RNNDataset
from src.month2.model_rnn import LSTMModel
from src.month2.preprocess import prepare_data # 내가 만든 전처리 함수

def train():
    # 설정값 불러오기
    with open('configs/month2_week1.yaml', 'r') as f:  # yaml 파일을 염
        config = yaml.safe_load(f) # yaml 파일을 딕셔너리 형태로 변환
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    # runs/ 관리
    save_path = f"runs/exp_{datetime.now().strftime('%m%d_%H%M')}" # 현재 시간을 해당 포멧의 문자로 변환
    os.makedirs(save_path, exist_ok = True) # 폴더를 만든다. (뒤에 키워드 전달인자는 이미 있어도 에러내지 말라는 시그널)
    
    # save_path/log.txt 주소를 생성
    log_file_path = os.path.join(save_path, "log.txt")
    
    # 논문 Section 3.1 : N 개의 토큰(t1, t2,...,tn)으로 이루어진 sequence을 생성
    numerical_data, vocab = prepare_data('data/harry_potter.txt') # 진짜 데이터 로드
    
    # dummy_data = list(range(1000)) : 임시 데이터 (나중에 해리포터 텍스트로 교체) 
    vocab_size = len(vocab.itos) # 단어장 크기를 모델에 전달하기 위해 계산

    dataset = RNNDataset(numerical_data, config['train']['seq_len']) # 중첩 딕셔너리 인덱싱
    
    # 데이터를 batch_size 크기만큼 묶어서 모델에게 전달해주는 역할을 하는 데이터 로더 생성
    dataloader = DataLoader(
        dataset, 
        batch_size = config['train']['batch_size'], 
        shuffle = True, 
        drop_last = True,
        num_workers = 4, # cpu 코어 4개를 써서 데이터를 미리 퍼옴, 디폴트는 0이고 GPU가 직접 데이터를 퍼와서 느림
        pin_memory = True # 메모리를 고정해서 GPU 전송 속도 향상
    )
    
    # 논문 Section 3.1 : L-Layer LSTM 모델 생성
    model = LSTMModel(
        vocab_size = vocab_size, 
        embed_dim = config['model']['embed_dim'], # 논문 Section 3.1 : Θ_x
        hidden_dim = config['model']['hidden_dim'], # 논문 Section 3.1 : Θ_LSTM
        num_layers = config['model']['num_layers'] # 논문 Section 3.1 : L_Layer
    ).to(device) # 모델을 GPU로 보냄
    
    # 오차 계산 도구 (확률을 높이는 것 == 오차(loss) 줄이는 것 <- 컴퓨터가 잘하는것)
    # 모델의 예측값과 실제 정답 사이의 오차 계산
    criterion = nn.CrossEntropyLoss(ignore_index = 0) # 패딩 토큰인 0번은 정답 체크에서 제외
    
    optimizer = torch.optim.Adam(model.parameters(), lr = config['train']['lr']) # 오차를 바탕으로 모델의 파라미터를 최적화해주는 optimizer

    print("Peters et al. 2018 BiLM Forward Training Start")
    
    # 진짜 학습 시작
    # 딥러닝에서 학습은 순전파(모델이 순수히 문제를 푸는 과정) + 역전파(틀린 만큼 오답노트하는 과정)
    for epoch in range(config['train']['epochs']):

        total_loss = 0

        for i, (x, y) in enumerate(dataloader): # batch의 입력값은 x, 정답은 y에 대입
            x, y = x.to(device), y.to(device) # x, y를 GPU로 이동
            
            # LSTM 모델의 장/단기 기억을 담고 있는 저장소인 hidden에 (h0, c0)대입
            # 배치가 섞여있으므로(shuffle = True), 매번 기억을 '리셋'하고 새로 시작해야 됨.
            hidden_f, hidden_b = model.init_hidden(config['train']['batch_size'], device) 
            optimizer.zero_grad() # 이전 계산값(기울기) 지움
            
            x_b = torch.flip(y, dims = [1]) # y(정답)를 뒤집은게 입력
            y_b = torch.flip(x, dims = [1]) # x(입력)를 뒤집은게 정답

            # 양방향 함수 호출
            logits_f, logits_b, hidden_f, hidden_b = model.forward_bi(x, x_b, hidden_f, hidden_b)
            
            # [Loss 계산, 순전파] 

            # 정방향 오차(forward pass)
            loss_f = criterion(logits_f.reshape(-1, vocab_size), y.reshape(-1)) # 시그마 log p(오차)를 계산

            # 역방향 오차(backward pass)
            loss_b = criterion(logits_b.reshape(-1, vocab_size), y_b.reshape(-1))  # 시그마 log p(오차)를 계산
            
            # 최종 오차 합산. 논문 Section 3.1 : maximizes the log likelihood of the forward and backward directions
            loss = loss_f + loss_b

            
            # [역전파] 오차를 줄이도록 가중치 업데이트    
            loss.backward()          
            optimizer.step()
            
                

        
        # 매 에폭의 결과를 log_msg 변수에 담음
        log_msg = f"Epoch {epoch + 1}: Loss {loss.item():.4f}"
        
        # save_path/log.txt 주소명을 열어서 log.txt파일 만들고 거기다가 log_msg 기록
        with open(log_file_path, "a") as f:
            f.write(log_msg + "\n")    
           
        print(f"Epoch {epoch + 1}: Loss {loss.item():.4f}") # loss 텐서에서 .item()으로 실수값만 리턴

    # save_path/model.pth 주소를 생성 후 model.pth 파일 생성 후 여기에 학습 종료 후 최종 모델 가중치 저장
    # save_path/vocab.pth 주소를 생성 후 vocab.pth 파일 생성 후 여기에 단어장도 함께 저장
    # 추론할때 두 파일 모두 필요
    torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))
    torch.save(vocab, os.path.join(save_path, "vocab.pth"))
    print(f"모든 결과가 {save_path}에 저장되었습니다!")    
        
"""
# 테스트 실행 코드   
if __name__ == "__main__":
    train()
"""



