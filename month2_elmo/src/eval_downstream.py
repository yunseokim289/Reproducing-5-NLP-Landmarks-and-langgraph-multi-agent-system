import torch
import numpy as np
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.month2.model_rnn import LSTMModel

def run_final_evaluation():
    # ==========================================
    exp_path = "runs/exp_0204_1348" 
    # ==========================================
    
    print(f">>> 🚀 모델 로딩 및 분석 시작... ({exp_path})")
    
    # 데이터 로드
    vocab = torch.load(os.path.join(exp_path, "vocab.pth"), weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 모델 설정 (Hidden Dim = 256 확인!)
    HIDDEN_DIM = 256 
    model = LSTMModel(len(vocab.itos), 128, HIDDEN_DIM, 2).to(device)
    model.load_state_dict(torch.load(os.path.join(exp_path, "model.pth"), map_location=device))
    model.eval()

    # [테스트 케이스] 단어 구성은 100% 같지만 뜻은 반대
    sent1 = "harry is good not bad"
    sent2 = "harry is bad not good"
    
    print(f"\n[분석 대상]")
    print(f"A: {sent1}")
    print(f"B: {sent2}")
    print("-" * 50)

    # 2. 벡터 추출
    tokens1 = sent1.split()
    tokens2 = sent2.split()
    idx1 = torch.tensor([[vocab.stoi.get(t, 0) for t in tokens1]]).to(device)
    idx2 = torch.tensor([[vocab.stoi.get(t, 0) for t in tokens2]]).to(device)

    with torch.no_grad():
        # Static Embedding (단어 평균)
        s1 = model.embedding(idx1).mean(dim=1)
        s2 = model.embedding(idx2).mean(dim=1)
        
        # ELMo Embedding (Bi-LSTM 문맥)
        # [수정 완료] 모델 크기에 맞춰서 256으로 설정!
        h_0 = (torch.zeros(2, 1, HIDDEN_DIM).to(device), 
               torch.zeros(2, 1, HIDDEN_DIM).to(device)) 
        
        # Forward & Backward Pass
        out_f1, _ = model.f_lstm(model.embedding(idx1), h_0)
        out_b1, _ = model.b_lstm(model.embedding(torch.flip(idx1, [1])), h_0)
        e1 = torch.cat((out_f1.mean(dim=1), out_b1.mean(dim=1)), dim=1)

        out_f2, _ = model.f_lstm(model.embedding(idx2), h_0)
        out_b2, _ = model.b_lstm(model.embedding(torch.flip(idx2, [1])), h_0)
        e2 = torch.cat((out_f2.mean(dim=1), out_b2.mean(dim=1)), dim=1)

    # 3. 수치 계산
    sim_static = F.cosine_similarity(s1, s2).item()
    sim_elmo = F.cosine_similarity(e1, e2).item()
    
    dist_static = 1.0 - sim_static
    dist_elmo = 1.0 - sim_elmo

    print(f"1️⃣  Static (단어 평균) | 유사도: {sim_static:.10f} | 거리: {dist_static:.10f}")
    print(f"2️⃣  ELMo   (Bi-LSTM)  | 유사도: {sim_elmo:.10f} | 거리: {dist_elmo:.10f}")
    
    # 4. 그래프 그리기
    print("\n>>> 🎨 그래프 그리는 중...")
    
    labels = ['Static\n(Word2Vec)', 'ELMo\n(Bi-LSTM)']
    
    plt.figure(figsize=(10, 5))
    
    # 왼쪽: 유사도
    plt.subplot(1, 2, 1)
    bars1 = plt.bar(labels, [sim_static, sim_elmo], color=['gray', '#d62728'], alpha=0.8, width=0.5)
    plt.title('Cosine Similarity (1.0 = Same)', fontsize=12, fontweight='bold')
    plt.ylim(0.95, 1.005)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                 f'{bar.get_height():.4f}', ha='center', va='bottom', fontweight='bold')

    # 오른쪽: 의미적 거리 (핵심!)
    plt.subplot(1, 2, 2)
    bars2 = plt.bar(labels, [dist_static, dist_elmo], color=['gray', '#1f77b4'], alpha=0.8, width=0.5)
    plt.title('Semantic Distance (Higher = Better)', fontsize=12, fontweight='bold')
    plt.ylim(0, max(dist_elmo * 1.5, 0.01)) 
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    for bar in bars2:
        h = bar.get_height()
        if h < 0.000001:
            plt.text(bar.get_x() + bar.get_width()/2, 0, 
                     "0.0 (Fail)", ha='center', va='bottom', color='red', fontweight='bold')
        else:
            plt.text(bar.get_x() + bar.get_width()/2, h, 
                     f'{h:.4f}', ha='center', va='bottom', color='blue', fontweight='bold')

    plt.tight_layout()
    plt.savefig('final_analysis.png')
    print(f">>> ✨ 저장 완료! 'final_analysis.png' 파일을 확인하세요.")

if __name__ == "__main__":
    run_final_evaluation()