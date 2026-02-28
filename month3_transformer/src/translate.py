import torch
import pickle
import json
import os
from modules import Transformer


# 1. 설정 파일 불러오기
with open('configs/base_config.json', 'r') as f:
    config = json.load(f) # 파일을 읽어서 딕셔너리 형태로 리턴

EXP_NAME = config['exp_name'] # 실험의 이름표(exp_name)을 추가    
D_MODEL = config['d_model']
N_HEAD = config['n_head']
D_FF = config['d_ff']
N_LAYER = config['n_layer']
MAX_LEN = config['max_len']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def translate_sentence(sentence, model, src_vocab, tgt_vocab, max_len=20):
    model.eval() # 평가 모드임을 모델에게 알려줌, DROPOUT 꺼짐
    
    # 1. 입력된 영어 문장을 숫자로 바꾸기
    tokens = sentence.lower().split()
    src_indices = src_vocab.encode(tokens)
    src_tensor = torch.tensor([src_indices]).to(DEVICE)

    tgt_indices = [tgt_vocab.stoi['<sos>']] # 실시간 번역 결과물에 시작신호 <sos>만 삽입

    # 3. 한 글자씩 예측하기 (Greedy Decoding)
    for i in range(max_len):
        tgt_tensor = torch.tensor([tgt_indices]).to(DEVICE)
        
        # 모델에게 지금까지의 상황을 주고 다음 단어 예측시킴
        with torch.no_grad(): # 지금부터 하는 계산은 기울기(Gradient) 추적하지 마!
            output = model(src_tensor, tgt_tensor) 
        
        # 첫 번째 문장(0)에서, 가장 마지막으로 예측한 단어 위치(-1)를 보고  처음부터 끝까지 모든 열을 리턴  
	    # argmax(): 가장 점수가 높은(Max) 단어의 위치 번호를 찾아라
        next_word_idx = output[0, -1, :].argmax().item()

        # 예측한 단어 번호를 실시간 번역 결과물에 추가
        tgt_indices.append(next_word_idx)

        # 만약 모델이 <eos> 끝을 외쳤다면 번역 종료
        if next_word_idx == tgt_vocab.stoi['<eos>']:
            break
    # 4. 예측한 숫자들을 다시 한글 단어로 바꾸기 
    # <sos>와 <eos>는 빼고 실제 단어만 가져옴
    translated_words = []
    for idx in tgt_indices:
        if idx not in [
            tgt_vocab.stoi['<sos>'],
            tgt_vocab.stoi['<eos>'],
            tgt_vocab.stoi['<pad>']
        ]:
            # 숫자에 해당하는 단어 찾기
            word = tgt_vocab.itos[idx]
            translated_words.append(word)

    return " ".join(translated_words)           


def main():

    # runs/ 관리
    save_dir = f'runs/{EXP_NAME}'

    # save_dir/log2.txt 주소를 생성
    log_file_path = os.path.join(save_dir, "log2.txt")

    with open(f'{save_dir}/vocab.pkl', 'rb') as f: # 'rb'는 피클(기계어) 읽어라
        vocabs = pickle.load(f) # 피클 데이터를 읽어서 딕셔너리 형태{'src':영어단어장, 'tgt':한글단어장}로 리턴 

    src_vocab = vocabs['src']
    tgt_vocab = vocabs['tgt']

    src_vocab_size = len(src_vocab.stoi)    
    tgt_vocab_size = len(tgt_vocab.stoi)
    
    # 모델 껍데기(아키텍처) 먼저 생성
    model = Transformer(
        vocab_size = max(src_vocab_size, tgt_vocab_size),
        d_model = D_MODEL,
        n_head = N_HEAD,
        d_ff = D_FF,
        max_len = MAX_LEN,
        n_layer = N_LAYER
    ).to(DEVICE)

    # 만들어둔 껍데기 모델에 학습때 저장해둔 파라미터를 덮어씌움
    model.load_state_dict(torch.load(f'{save_dir}/transformer_weights.pth', map_location = DEVICE))

    print("번역기 준비 완료 (종료하려면 'q'를 입력하세요)")

    while True:
        
        # 사용자 입력받기
        sentence = input("영어 문장 입력: ")
        
        if sentence.lower() == 'q':
            print("번역기를 종료합니다. 수고하셨습니다.")
            log_msg = f"한글 번역 결과: {translation}"
            # save_path/log.txt 주소명을 열어서 log_msg 기록
            with open(log_file_path, "a") as f:
                f.write(log_msg + "\n")
            break

        try:
            translation = translate_sentence(sentence, model, src_vocab, tgt_vocab, MAX_LEN)    

            print(f"한글 번역 결과: {translation}")
            log_msg = f"한글 번역 결과: {translation}"
        except Exception as e:
            print(f"예외 발생: 사전에 없는 단어이거나 너무 깁니다. ({e})")
            log_msg = f"한글 번역 결과: {translation}"
        
        # save_path/log.txt 주소명을 열어서 log_msg 기록
        with open(log_file_path, "a") as f:
            f.write(log_msg + "\n")

"""
# 테스트 실행 코드
if __name__ == "__main__":
    main()

"""
