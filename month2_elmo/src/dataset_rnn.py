import torch
from torch.utils.data import Dataset # 파이토치의 표준 데이터 양식을 불러옴

# 해리포터 텍스트를 학습하기 좋은 형태로 자름
class RNNDataset(Dataset):
    def __init__(self, data_indices, seq_len):
        self.data = data_indices # 숫자로 바뀐 해리포터 텍스트 전체
        self.seq_len = seq_len # 논문에서 설정한 한 번에 읽을 단어 길이

    def __len__(self):
        # 전체 데이터 길이에서 seq_len만큼 뺀 값이 우리가 만들 수 있는 샘플의 총 개수
        return len(self.data)-self.seq_len

    def __getitem__(self, idx):
        # 파이토치로 리스트를 계산하기 위해 '텐서'형태로 변환
        x = torch.tensor(self.data[idx : idx + self.seq_len]) # x는 현재의 입력(예 : 0번~29번)
        y = torch.tensor(self.data[idx + 1 : idx + self.seq_len + 1]) # y는 정답(예 : 1번~30번)
        
        return x, y


             