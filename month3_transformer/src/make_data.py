import os

def create_dummy_data():
    # 1. 영어 문장(source) 리스트 만들기
    src_sentences = [
        "i love you",
        "hello world",
        "good morning",
        "how are you",
        "i am a student",
        "deep learning is fun",
        "thank you very much",
        "see you later",
        "what is your name",
        "i like harry potter"
    ] * 100

    # 2. 한글 문장(target) 리스트 만들기
    tgt_sentences = [
        "나는 너를 사랑해",
        "안녕 세상아",
        "좋은 아침",
        "잘 지내니",
        "나는 학생이다",
        "딥러닝은 재밌어",
        "정말 고마워",
        "나중에 봐",
        "너의 이름은 뭐니",
        "나는 해리포터를 좋아해"    
    ] * 100

    # 3. 폴더 만들기
    if not os.path.exists('data'):
        os.makedirs('data') # 'data'란 이름의 폴더가 없으면 만들어라

    # 4. 파일 저장하기(영어)
    with open('data/src.txt','w',encoding = 'utf-8') as f:
        for line in src_sentences:
            f.write(line + '\n') 

    # 5. 파일 저장하기(한글)
    with open('data/tgt.txt','w',encoding = 'utf-8') as f:
        for line in tgt_sentences:
            f.write(line + '\n')        

    print("데이터 생성 완료!")        


"""
# 테스트 실행 코드
if __name__ == "__main__":
    create_dummy_data()    
    
"""