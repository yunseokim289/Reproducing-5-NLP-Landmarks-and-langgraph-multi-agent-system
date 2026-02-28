import torch
import json
import pickle
import random
from nltk.translate.bleu_score import corpus_bleu
from modules import Transformer

# ==========================================
# 1. 설정 및 모델/단어장 불러오기
# ==========================================
with open('configs/base_config.json', 'r') as f:
    config = json.load(f)

EXP_NAME = config['exp_name']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = f'runs/{EXP_NAME}'

with open(f'{SAVE_DIR}/vocab.pkl', 'rb') as f:
    vocabs = pickle.load(f)
src_vocab = vocabs['src']
tgt_vocab = vocabs['tgt']

model = Transformer(
    vocab_size=max(len(src_vocab.stoi), len(tgt_vocab.stoi)),
    d_model=config['d_model'],
    n_head=config['n_head'],
    d_ff=config['d_ff'],
    max_len=config['max_len'],
    n_layer=config['n_layer']
).to(DEVICE)

model.load_state_dict(torch.load(f'{SAVE_DIR}/transformer_weights.pth', map_location=DEVICE))
model.eval()

# ==========================================
# 2. 번역 함수 (BLEU 채점용으로 살짝 수정)
# ==========================================
def translate_sentence_for_bleu(sentence, model, src_vocab, tgt_vocab, max_len=50):
    tokens = sentence.lower().split()
    src_indices = [src_vocab.stoi.get('<sos>')] + \
                  [src_vocab.stoi.get(token, src_vocab.stoi.get('<unk>')) for token in tokens] + \
                  [src_vocab.stoi.get('<eos>')]
    src_tensor = torch.tensor([src_indices]).to(DEVICE)
    
    tgt_indices = [tgt_vocab.stoi.get('<sos>')]
    
    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_indices]).to(DEVICE)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)
        
        next_word_idx = output[0, -1, :].argmax().item()
        tgt_indices.append(next_word_idx)
        
        if next_word_idx == tgt_vocab.stoi.get('<eos>'):
            break
            
    # 채점을 위해 문장(String)이 아니라 단어 리스트(List) 형태로 반환!
    translated_words = []
    for idx in tgt_indices:
        if idx not in [tgt_vocab.stoi.get('<sos>'), tgt_vocab.stoi.get('<eos>'), tgt_vocab.stoi.get('<pad>')]:
            translated_words.append(tgt_vocab.itos[idx])
            
    return translated_words

# ==========================================
# 3. 메인 채점 로직
# ==========================================
def main():
    print("📚 원본 데이터(정답지)를 불러오는 중...")
    with open('data/src.txt', 'r', encoding='utf-8') as f:
        src_lines = f.read().splitlines()
    with open('data/tgt.txt', 'r', encoding='utf-8') as f:
        tgt_lines = f.read().splitlines()

    # 무작위로 100문제만 출제!
    sample_size = 100
    indices = random.sample(range(len(src_lines)), sample_size)
    
    actuals = []     # 사람이 번역한 진짜 정답 (Reference)
    predictions = [] # 모델이 번역한 제출 답안 (Candidate)

    print(f"\n📝 무작위 {sample_size}개 문장 번역 및 BLEU 평가 시작...")
    
    for i, idx in enumerate(indices):
        src_sentence = src_lines[idx]
        tgt_sentence = tgt_lines[idx]
        
        # 정답지 준비: NLTK는 정답이 여러 개일 수 있다고 가정하므로 이중 리스트로 묶어줍니다.
        actual_tokens = tgt_sentence.split()
        actuals.append([actual_tokens]) 
        
        # 모델의 답안지 준비
        pred_tokens = translate_sentence_for_bleu(src_sentence, model, src_vocab, tgt_vocab)
        predictions.append(pred_tokens)
        
        # 진행 상황 출력
        if (i + 1) % 20 == 0:
            print(f"⏳ [{i+1}/{sample_size}] 번역 완료...")

    # NLTK를 이용해 BLEU Score 계산 (0 ~ 100점)
    bleu_score = corpus_bleu(actuals, predictions) * 100
    
    print("\n" + "="*45)
    print(f"🎉 최종 모의고사 BLEU Score: {bleu_score:.2f} 점 / 100.00 점")
    print("="*45)

if __name__ == "__main__":
    main()