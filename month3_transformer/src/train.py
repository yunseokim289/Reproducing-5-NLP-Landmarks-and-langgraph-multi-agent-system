import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import os
import pickle
import json

from modules import Transformer
from preprocess import Vocab, prepare_data
from dataset import TranslationDataset

# 1. 설정 파일 불러오기
with open('configs/base_config.json', 'r') as f:
    config = json.load(f) # 파일을 읽어서 딕셔너리 형태로 리턴

EXP_NAME = config['exp_name'] # 실험의 이름표(exp_name)을 추가    
BATCH_SIZE = config['batch_size']
D_MODEL = config['d_model']
N_HEAD = config['n_head']
D_FF = config['d_ff']
N_LAYER = config['n_layer']
MAX_LEN = config['max_len']
EPOCHS = config['epochs']
LEARNING_RATE = config['learning_rate']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print(f"학습 시작! 사용 장치:{DEVICE}")

    # 2. 단어장(Vocab) 만들기
    with open('data/src.txt', 'r', encoding = 'utf-8') as f:
        src_tokens = f.read().split()

    with open('data/tgt.txt', 'r', encoding = 'utf-8') as f:
        tgt_tokens = f.read().split()

    # 단어장 생성 (min_freq = 2로 설정해서 2번 이상 등장한 단어만 학습하도록 함)
    src_vocab = Vocab(src_tokens, min_freq = 2)
    tgt_vocab = Vocab(tgt_tokens, min_freq = 2)

    src_vocab_size = len(src_vocab.stoi)  
    tgt_vocab_size = len(tgt_vocab.stoi)

    print(f"영어 단어 수: {src_vocab_size}")
    print(f"한글 단어 수: {tgt_vocab_size}")  
    

    # runs/ 관리
    save_dir = f'runs/{EXP_NAME}'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 

    # save_path/log.txt 주소를 생성
    log_file_path = os.path.join(save_dir, "log.txt")

    # 3. 데이터 로더 준비
    dataset = TranslationDataset(
        'data/src.txt', 'data/tgt.txt',
        src_vocab, tgt_vocab,
        max_len = MAX_LEN
    )

    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

    # 4. 모델 준비

    model = Transformer(
        vocab_size = max(src_vocab_size, tgt_vocab_size),
        d_model = D_MODEL,
        n_head = N_HEAD,
        d_ff = D_FF,
        max_len = MAX_LEN,
        n_layer = N_LAYER
    ).to(DEVICE)

    # 손실 함수(Loss Fuction)
    # ignore_index = 0: 패딩(0) 부분은 틀려도 점수 깎지 말것
    criterion = nn.CrossEntropyLoss(ignore_index = 0)

    # 최적화 도구 (Optimizer)
    optimizer = Adam(model.parameters(), lr = LEARNING_RATE)

    # 5. 학습 루트

    model.train() # 학습 모드임을 모델에게 알려줌, Dropout 켜짐

    for epoch in range(EPOCHS):
        total_loss = 0

        for batch_idx, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)

            # ★ 타겟 데이터 자르기 (Teacher Forcing) <- 암기
            # 모델에게 보여주는 입력 (Input): 마지막 <eos> 빼기
            # 모델이 맞춰야 할 정답 (Label): 첫번째 <sos> 빼기    
             
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # 예측값(logits) 구하기 (forward)
            logits = model(src, tgt_input)

            # 오차(Loss) 구하기

            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))

            # 역전파(backward)
            optimizer.zero_grad() # 지난번 학습때 계산 결과 청소
            loss.backward() # 오차 역전파: 각 가중치가 틀린 만큼의 '기울기' 계산
            optimizer.step() # 가중치 업데이트: 기울기 방향으로 모델 수정

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader) # 에폭마다 평균 오차 계산(batch개수로 나눔)

        # 매 에폭(턴)이 끝날 때마다 그냥 정직하게 다 출력하기
        print(f"현재 에폭: {epoch + 1}, 총 에폭: {EPOCHS}, 평균 오차: {avg_loss}")
        # 매 에폭의 결과를 log_msg 변수에 담음
        log_msg = f"현재 에폭: {epoch + 1}, 총 에폭: {EPOCHS}, 평균 오차: {avg_loss}"
        
        # save_path/log.txt 주소명을 열어서 log_msg 기록
        with open(log_file_path, "a") as f:
            f.write(log_msg + "\n")
    

    """
    state_dict() : 모델의 구조(뼈대)는 제외하고, 모델이 학습을 통해 업데이트된 
    '가중치(Weight)와 편향(Bias)' 숫자들만 딕셔너리(Key-Value) 형태로 추출한 뒤,
    추출한 데이터를 하드디스크에 .pth란 확장자로 저장
    """
    torch.save(model.state_dict(), f'{save_dir}/transformer_weights.pth')
    
    # vocab.pkl 파일 생성후 기계여(pickle)로 작성
    with open(f'{save_dir}/vocab.pkl','wb') as f:
        pickle.dump({'src': src_vocab, 'tgt': tgt_vocab}, f) # 딕셔너리를 f에 던져넣음
    
    print("학습 완료! 모델과 단어장이 runs/ 폴더에 저장되었습니다.")
    


# 테스트 실행 코드
if __name__ == "__main__":
    main()

