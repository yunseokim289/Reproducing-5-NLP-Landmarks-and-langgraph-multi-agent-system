import torch
import yaml
import os
from datetime import datetime
from src.month2.model_rnn import LSTMModel
from src.month2.preprocess import Vocab # 저장된 단어장을 읽기 위해 불러옴

def generate():
    # 설정값 및 장치 설정
    with open('configs/month2_week1.yaml') as f:
        config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # runs/ 관리
    
    exp_path = "runs/exp_0204_1348" # 실제 학습 결과가 저장된 폴더 경로를 저장

    # exp_path/inference.txt 주소를 생성
    inference_file_path = os.path.join(exp_path, "inference2.txt")

    model_path = os.path.join(exp_path, "model.pth")  # exp_path/model.pth 주소 생성 후 저장
    vocab_path = os.path.join(exp_path, "vocab.pth")  # exp_path/vocab.pth 주소 생성 후 저장
    
    vocab = torch.load(vocab_path, weights_only = False) # 해당 경로에 있는 단어장 (텐서)를 불러오기, 보안 경고를 무시하고 염
    vocab_size = len(vocab.itos) # 단어장의 크기를 저장

    # 모델(틀) 다시 생성 (학습 때와 동일한 규격이어야 됨)
    model = LSTMModel(
        vocab_size = vocab_size, # 실제 단어장 크기 대입
        embed_dim = config['model']['embed_dim'], # 논문 Section 3.1 : Θ_x
        hidden_dim = config['model']['hidden_dim'], # 논문 Section 3.1 : Θ_LSTM
        num_layers = config['model']['num_layers'] # 논문 Section 3.1 : L_Layer
    ).to(device) # 모델을 GPU로 보냄

    # 저장된 학습 후 모델(파라미터) 불러오기
    # torch.load로 exp_path/model.pth 이 주소의 model.pth(학습 후 모델의 파라미터) 파일을 불러와서 빈 모델의 각 층에 끼워넣음
    model.load_state_dict(torch.load(model_path, map_location = device))
    
    model.eval() # 모델을 평가(eval) 모드로 전환하여 지금은 학습이 아닌 평가 단계임을 모델에게 알려줌
    
    print(f"모델 로드 완료: {model_path}")

    # 입력 데이터 준비
    # 시작 단어를 실제 단어 'harry'로 설정
    start_word = 'harry'
    input_idx = vocab.stoi.get(start_word, vocab.stoi['<unk>']) # get(단어, 기본값) : 단어가 사전에 없으면 1번을 가져오라는 안전장치

    # 파이토치 모델은 [배치크기, 문장길이] 형태의 '2차원 텐서'를 입력으로 받음
    input_seq = torch.tensor([[input_idx]]).to(device)

    hidden, _ = model.init_hidden(1, device)
    
    print(f"{start_word}", end = " ")

    # 시작 단어를 inference_msg 변수에 담음
    inference_msg = f"{start_word}" + " "

    # exp_path/log2.txt 주소를 열어서 log2.txt파일 생성 후 시작단어를 기록
    with open(inference_file_path, "a", encoding = "utf-8") as f:
        f.write(inference_msg) 

    # 다음 단어를 예측
    # 추론 단계에서는 답만 내면 되기 때문에, with안에 들어온 순간 지금부터 하는 계산은 메모리에 기록하지 말라고 명령

    with torch.no_grad():
        # 모델에 입력값과 초기 기억을 넣고 결과를 받음
        # 이때 계산을 메모리에 기록하지 않으므로 속도가 빨라지고 메모리 사용량이 확 줄어듦
        for _ in range(50): # 총 50개의 단어를 하나씩 이어 붙일 때까지 반복
        
            output, hidden = model(input_seq, hidden) # 결과물(모든 단어에 대한 점수판)과 입력에 대한 기억을 저장
        
            # 모든 단어장에 있는 단어 후보 중 입력 번호 다음으로 올 점수 중 최댓값이 위치한 인덱스(번호) 찾음
            # {3주차 greedy algorithm} predicted_idx = torch.argmax(output, dim = 2).item() 
            
            # {4주차: Sampling(확률에 따라 단어 뽑기)} 
            # 온도를 적용하여 점수를 조절
            # temperature이 낮으면 1등 점수가 압도적으로 커지고, 높으면 점수들이 다 비슷비슷해짐.
            temperature = 0.8 # 0.1~0.5 사이로 조절 가능
     
            top_k = 50 # TOP-K 필터링 로직 추가. 상위 50개 단어만 후보로 남김.

            logits = output / temperature

            top_k_values, top_k_indices = torch.topk(logits, top_k) # logits 중에서 상위 k개의 값과 그 위치(인덱스)를 리턴

            min_values = torch.full_like(logits, float('-inf')) # logits을 마이너스 무한대로 다 채워서 리턴
            
            # min_values에다가 해당 위치에 해당 값을 뿌린다.
            # 상위 k개가 아닌 값들은 전부 -무한대로 채워지게 됨. (top-k 필터링)
            logits = min_values.scatter(2, top_k_indices, top_k_values) 

            top_p = 0.9 # TOP-P 필터링 로직 추가. 누적 확률 90%까지의 단어만 살리겠다는 뜻

            sorted_logits, sorted_indices = torch.sort(logits, descending = True, dim = -1) # logits을 내림차순 정렬

            cumulative_probs = torch.softmax(sorted_logits, dim = -1).cumsum(dim = -1) # top-k 필터링된 원점수(logits)을 0~1로 변환 후 앞에서부터 차례대로 합하기 (누적 합계)
            
            sorted_indices_to_remove = cumulative_probs > top_p # 누적 확률이 0.9를 넘어가는 단어들(True : 탈락 후보들) 찾아내기
            
            # 1등 단어는 확률이 0.9를 넘어도 절대 버리지 않기 위해 한 칸씩 뒤로 밀기
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False # 1등은 무조건 살려둠
            
            # 내림차순 정렬된 인덱스에 탈락딱지(True,False)을 뿌린다.
            indices_to_remove = sorted_indices_to_remove.scatter(2, sorted_indices, sorted_indices_to_remove)
            
            # logits(숫자판)에 indices_to_remove(탈락딱지판)을 겹쳐서 True인 점수에만 -무한대를 준다. (top-p 필터링)
            logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(logits, dim = 2) # top-k, top-p 이중 필터링된 원점수(logits)을 0~1로 변환
            
            """
            multinomial()은 입력받은 확률(probs)에 따라 랜덤하게 단어 하나(인덱스)를 뽑음.
            예: 70% 확률인 단어는 70% 빈도로, 1% 확률인 단어는 1%의 빈도로 뽑힘. 
            결과물은 텐서
            """
            
            predicted_idx = torch.multinomial(probs.squeeze(), num_samples = 1).item()

            # 번호를 다시 사람이 읽을 수 있는 단어로 바꿈
            predicted_word = vocab.itos[predicted_idx] 

            # 단어를 화면에 출력
            print(predicted_word, end = " ")
            
            # 추론 결과를 log_msg 변수에 담음
            inference_msg = f"{predicted_word}" + " "

            # exp_path/log2.txt 주소를 열어서 log2.txt파일 생성 후 추론 결과를 기록
            with open(inference_file_path, "a", encoding = "utf-8") as f:
                f.write(inference_msg) 

            # 방금 예측한 단어 번호를 다음 입력값으로 업데이트
            input_seq = torch.tensor([[predicted_idx]]).to(device)
            

"""
# 테스트 실행 코드
if __name__ == "__main__":
    generate()
"""



