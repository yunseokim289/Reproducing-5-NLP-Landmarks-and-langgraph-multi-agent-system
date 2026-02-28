import zipfile
import os

print("진짜 말뭉치 다운로드 시작")

# 1. 파일 다운로드 (ManyThings 오픈소스 데이터)
url = "http://www.manythings.org/anki/kor-eng.zip"

# 해당 url로 들어가서 파일을 내 폴더에 "kor-eng.zip"란 이름으로 저장해줘!
os.system("wget http://www.manythings.org/anki/kor-eng.zip")


# 2. 압축 풀기
with zipfile.ZipFile("kor-eng.zip", 'r') as zip_ref: # .zip 압축 파일 풀기
    zip_ref.extractall("data/") # 압축 파일 안에 있는 모든 내용을 "data/" 폴더 안에 풀어버려!

# 3. 영어(src)와 한글(tgt) 분리해서 저장하기

src_lines, tgt_lines = [], []
with open("data/kor.txt", "r", encoding = "utf-8") as f:
    for line in f:
        parts = line.split('\t')

        if len(parts) >= 2:
            src_lines.append(parts[0].strip().lower()) # 모델은 A == a로 인식

            tgt_lines.append(parts[1].strip()) # 한글은 대소문자가 없으므로 strip()만 호출


# 4. 파일로 쓰기 (기존의 src.txt와 tgt.txt을 전부 지우고 새로 작성)
with open("data/src.txt", "w", encoding = "utf-8") as f_src, open ("data/tgt.txt", "w", encoding = "utf-8") as f_tgt:
    f_src.write("\n".join(src_lines)) 
    f_tgt.write("\n".join(tgt_lines))   


# 5. 찌꺼기 파일 삭제
os.remove("kor-eng.zip")  
os.remove("data/kor.txt") 

print(f"성공! data/ 폴더에 {len(src_lines)}문장의 src.txt와 tgt.txt가 준비되었습니다.")
