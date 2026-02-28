# 터미널에 chmod +x ./scripts/gen_rnn.sh 입력해서 권한 부여
# 터미널에 ./scripts/gen_rnn.sh 입력해서 실행버튼 실행
# 현재 폴더(.)를 파이썬이 파일을 찾는 경로(PYTHONPATH)에 추가합니다.
export PYTHONPATH=$PYTHONPATH:.

python src/month2/gen_rnn.py