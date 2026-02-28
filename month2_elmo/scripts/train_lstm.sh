# 터미널에 chmod +x ./scripts/train_lstm.sh 입력해서 권한 부여
# 터미널에 ./scripts/train_lstm.sh 입력해서 실행버튼 실행

export PYTHONPATH=$PYTHONPATH:. # 현재 폴더(.)를 파이썬이 파일을 찾는 경로(PYTHONPATH)에 추가합니다.

python src/month2/train_rnn.py --config configs/month2_week1.yaml # configs에 정의된 하이퍼파라미터를 읽어서 학습 엔진을 실행합니다.