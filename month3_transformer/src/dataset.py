import torch
from torch.utils.data import Dataset 

class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_vocab, tgt_vocab, max_len = 100):
        # 1. 파일 읽어오기
        with open(src_file, 'r', encoding = 'utf-8') as f:
            # .read(): 파일 전체를 통째로 읽음
            # .splitlines(): 엔터키(\n) 기준으로 잘라서 리스트로 만듦
            self.src_lines = f.read().splitlines() 

        # 1. 파일 읽어오기
        with open(tgt_file, 'r', encoding = 'utf-8') as f:
            # .read(): 파일 전체를 통째로 읽음
            # .splitlines(): 엔터키(\n) 기준으로 잘라서 리스트로 만듦
            self.tgt_lines = f.read().splitlines()     

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab      
        self.max_len = max_len

    
    # 데이터가 총 몇개냐고 물어볼 때 이를 대답하는 함수
    def __len__(self):
        return len(self.src_lines)

    # ★ n번째 데이터 달라고 할때 실행되는 함수 
    def __getitem__(self, idx):

        # 1. 문장 꺼내기(idx번째)
        src_text = self.src_lines[idx]
        tgt_text = self.tgt_lines[idx]   

        # 2. 숫자로 변환 
        src_indices = self.src_vocab.encode(src_text.split())
        tgt_indices = self.tgt_vocab.encode(tgt_text.split())
        
        # 작문 문장을 [시작 + 내용 + 끝] 형태로 만듦
        tgt_indices = [self.tgt_vocab.stoi['<sos>']] + tgt_indices + [self.tgt_vocab.stoi['<eos>']]
        
        # 길이 맞추는 함수
        def pad_sequence(indices, max_len):
            # 너무 길면 자름
            if len(indices) > max_len:
                indices = indices[:max_len]

            # 너무 짧으면 뒤에 0을 붙임
            return torch.tensor(indices + [0] * (max_len - len(indices)))
        
        return pad_sequence(src_indices, self.max_len), pad_sequence(tgt_indices, self.max_len)


