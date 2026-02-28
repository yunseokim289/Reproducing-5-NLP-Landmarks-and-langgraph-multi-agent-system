import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast # RTX 4090 가속용 도구
from model import SkipGramModel, NegativeSamplingLoss # step 3 구현체
from dataset import Word2VecDataset # step 2 구현체

# RTX 4090 최적화을 위한 설정값(하이퍼파라미터)
EMB_DIM = 100 # 단어 벡터 차원
BATCH_SIZE = 4096 # 배치 크기 (넉넉히)
LEARNING_RATE = 0.001 # 학습률
EPOCHS = 5 # 전체 데이터 반복 횟수
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 최종 결정된 장치 정보

def train():
    # 데이터 로드
    dataset = Word2VecDataset("data/toy_data.txt")
    # 모델에 데이터를 효율적으로 공급하는 컨베이어 벨트 DataLoader 생성
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
    vocab_size = len(dataset.vocab) # 단어장의 길이 저장
    
    # 모델과 손실함수를 위에서 정한 장치 메모리로 이동
    model = SkipGramModel(vocab_size, EMB_DIM).to(DEVICE) 
    criterion = NegativeSamplingLoss().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE) # 최적화 알고리즘 Adam 생성
    scaler = GradScaler() # 스케일러 선언: FP16 연산 중 숫자가 너무 작아져서 사라지는 것(Underflow)을 방지합니다.

    print(f"학습 시작! 장치 : {DEVICE} | 데이터 쌍 : {len(dataset)}개")

    for epoch in range(EPOCHS):
        model.train() # 모델을 학습 모드로 전환
        total_loss = 0

        for i, (center, pos, neg) in enumerate(dataloader):
            center, pos, neg = center.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
            
            optimizer.zero_grad() # 이전 루프의 기울기 잔상이 남지 않게 0으로 초기화
            
            """
            AMP의 핵심 개념 (Mixed Precision)
            딥러닝 연산은 보통 FP32(32-bit Floating Point)라는 아주 정밀한 숫자를 사용합니다. 하지만 모든 연산에 이렇게 높은 정밀도가 필요한 것은 아닙니다.

            FP32 (Full Precision): 아주 정확하지만 메모리를 많이 먹고 계산이 느립니다.

            FP16 (Half Precision): 정확도는 조금 낮지만 메모리를 절반만 쓰고 계산 속도가 훨씬 빠릅니다.

            AMP: 이 둘을 자동(Automatic)으로 섞어줍니다.
            중요한 가중치 업데이트는 FP32로, 단순한 행렬 곱셈은 FP16으로 처리하죠.

            아래 두 문단은 AMP를 조절하는 역할을 수행
            """

            # autocast 구역: 이 안에서 일어나는 모델 연산은 자동으로 FP16으로 변환됩니다.
            with autocast():
                pos_score, neg_score = model(center, pos, neg)
                loss = criterion(pos_score, neg_score)
            
            scaler.scale(loss).backward() # 오차 역전파: 각 가중치가 틀린 만큼의 '기울기' 계산
            scaler.step(optimizer)        # 가중치 업데이트: 기울기 방향으로 모델 수정
            scaler.update()               # 스케일러 갱신

            total_loss += loss.item() # 텐서 내부의 순수 숫자(스칼라)만 추출

            if i % 100 == 0: # 중간 보고용 테스트 코드
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Step [{i}/{len(dataloader)}] | Loss: {loss.item():.4f}")
            
            # 에포크 당 평균 손실 계산  (전체 손실 / 총 묶음 수)
            # len(dataloader) : 224만 개 데이터를 4,096개씩 묶었을 때 나오는 묶음(Batch)의 총 개수
            print(f"Epoch {epoch+1} 완료! 평균 Loss : {total_loss / len(dataloader) :.4f}")    
            

            """
            state_dict() : 모델의 구조(뼈대)는 제외하고, 모델이 학습을 통해 업데이트된 
            '가중치(Weight)와 편향(Bias)' 숫자들만 딕셔너리(Key-Value) 형태로 추출한 뒤,
            추출한 데이터를 하드디스크에 .pth란 확장자로 저장
            
            """
            torch.save(model.state_dict(), "word2vect_model.pth")
            print("모델 저장 완료 : word2vect_model.pth")


# 테스트 실행 코드            
if __name__ == "__main__":
    train()


            

