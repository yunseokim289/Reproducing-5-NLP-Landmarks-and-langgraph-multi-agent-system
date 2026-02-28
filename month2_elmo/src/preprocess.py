import os
import re # 정규포현식 하는 모듈
from collections import Counter # 리스트 내 요소들의 개수를 세서 딕셔너리 형태로 변환

class Vocab:
    def __init__(self, tokens, min_freq = 2):

        counter = Counter(tokens) # 단어(토큰)별 빈도수를 계산하여 딕셔너리 형태로 변환

        self.itos = ['<pad>', '<unk>'] # 인덱스 0번은 패딩(길이 맞춤), 1번은 모르는 단어용으로 미리 지정

        self.itos += [word for word, count in counter.items() if count >= min_freq]

        self.stoi = {word : i for i, word in enumerate(self.itos)} # 딕셔너리 컴프리핸션

    def encode(self, tokens):
        numerical_list = [] # 숫자 담을 빈 리스트

        for token in tokens:
            if token in self.stoi:
                # 아는 단어라면? 단어장에서 그 번호를 찾아 바구니에 넣음
                idx = self.stoi[token]
                numerical_list.append(idx)

            else:
                # 처음 보는 단어면? <unk> 토큰의 번호(1번)을 대신 넣음
                unk_idx = self.stoi['<unk>']
                numerical_list.append(unk_idx)

        return numerical_list # 완성된 숫자 리스트 리턴

def clean_text(text):
    text = text.lower() # AI는 'Harry'와 'harry'를 다른 단어로 인식하기 때문에 '소문자'로 통일
        
    """
    정규표현식 문법 :
    re.sub(패턴, 바꿀문자, 대상)
    r'[^a-z\s]' -> 영문 소문자(a-z)와 공백(\s)이 아닌(^) 모든 것
    즉, 숫자, 마침표, 쉼표, 특수기호를 다 찾아서 빈칸으로 바꿔서 지워버림
    """
    text = re.sub(r'[^a-z\s]', ' ', text)

    return text.strip() # 문장 맨 앞과 맨 뒤에 붙은 불필요한 공백을 제거 후 리턴


def prepare_data(file_path):
    if not os.path.exists(file_path):
        print(f"에러 : {file_path} 파일이 없습니다!")
        return None, None
        
    # 파일을 읽기 모드로 염. utf-8은 한글/특수문자 깨짐 방지용
    f = open(file_path, 'r', encoding = 'utf-8')              
    raw_text = f.read() # 파일의 모든 글자를 읽어 저장
    f.close() # 파일을 다 읽었으니 닫아줌

    cleaned_text = clean_text(raw_text) # 글자를 깨끗이 정제
    tokens = cleaned_text.split()

    vocab = Vocab(tokens, min_freq = 2) # 단어장 생성(최소 2번은 나온 단어만 삽입)

    numerical_data = vocab.encode(tokens) # 단어 리스트(["harry", ...])를 숫자 리스트([2, ...])로 바꿈
        
    print("전처리 완료! (harry_potter.txt 기준)")
    print(f"총 단어 수 : {len(tokens)}")
    print(f"단어장 크기 : {len(vocab.itos)}")
        
    return numerical_data, vocab

"""
테스크 실행 코드
if __name__ == "__main__":
    data, vocab = prepare_data('data/harry_potter.txt')
"""


