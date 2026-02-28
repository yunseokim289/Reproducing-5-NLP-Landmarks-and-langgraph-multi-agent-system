import torch
import torch.nn.functional as F
from src.dataset import Word2VecDataset

def run_test():

    dataset = Word2VecDataset("data/toy_data.txt")
    
    checkpoint = torch.load("word2vect_model.pth", map_location = "cpu") # RTX 4090(GPU)에서 학습된 모델을 CPU 메모리로 안전하게 복사하여 로드
    embeddings = checkpoint['in_embed.weight'] # 모델 가중치 중 실제 단어 벡터가 담긴 부분만 추출

    
    # 기능 1 : 특정 단어와 가장 닮은(유사한) 단어 찾기
    def find_most_similar(query_word, top_k = 5):
        # 입력한 단어가 단어장에 있는지 먼저 확인
        if query_word not in dataset.word2idx:
            print(f"{query_word}단어는 단어장에 없습니다.")
            return 

        query_idx = dataset.word2idx[query_word]
        query_vec = embeddings[query_idx]

        # 모든 단어와의 점수를 담을 리스트
        all_scores = []

        # 단어장을 하나씩 훑으면서 점수를 계산
        for i in range(len(dataset.idx2word)):
            target_vec = embeddings[i]

            """
            두 벡터 사이의 '코사인 유사도'를 계산
            수식 : (A와 B의 내적) / (A의 길이 * B의 길이)
            """
            similarity = torch.dot(query_vec, target_vec) / (torch.norm(query_vec) * torch.norm(target_vec))
            
            # (단어이름, 점수) 쌍으로 저장
            all_scores.append((dataset.idx2word[i], similarity.item()))
            
        # 점수가 높은 순서대로 정렬 (내림차순)
        all_scores.sort(key = lambda x : x[1], reverse = True)

        print(f"{query_word}와 가장 유사한 단어들:")
           
            
        # 전체 리스트 길이에서 자기 자신(1개)을 뺀 값과 top_k 중 작은 것을 선택합니다.
        actual_top_k = min(top_k, len(all_scores) - 1)

        # 뽑을 수 있는 개수(actual_top_k)만큼만 반복문을 돕니다.
        for i in range(1, actual_top_k + 1):
            name, score = all_scores[i]
            print(f"- {name}: {score:.4f}")
            
        if actual_top_k == 0:
                print("유사한 단어를 더 이상 찾을 수 없습니다. (단어장 크기 부족)")

    # 기능 2 단어 유추 테스트(Analogy Evaluation)           
    def solve_analogy(a, b, c, top_k = 1):
        """
        단어 사이의 관계를 산술 연산으로 푸는 기능
        예: 'man' : 'king' = 'woman' : '?' -> 정답: 'queen'
        원리: king 벡터에서 man의 성질을 빼고 woman의 성질을 더하면 queen의 위치가 나온다.
        """
        for word in [a, b, c]:
            if word not in dataset.word2idx:
                print(f"{word} 단어가 단어장에 없어 유추가 불가능합니다.")
                return

        # 벡터 연산 (King - Man + Woman)
        vec_a = embeddings[dataset.word2idx[a]]
        vec_b = embeddings[dataset.word2idx[b]]
        vec_c = embeddings[dataset.word2idx[c]] 
        target_vec = vec_b - vec_a + vec_c

        # 모든 단어와의 점수를 담을 리스트
        all_scores = []
        for i in range(len(dataset.idx2word)):
            target_word = dataset.idx2word[i]
            # 입력값 (a,b,c)와 똑같은 단어는 제외
            if target_word in [a, b, c]:
                continue

            current_vec = embeddings[i]

            """
            두 벡터 사이의 '코사인 유사도'를 계산
            수식 : (A와 B의 내적) / (A의 길이 * B의 길이)
            """
            similarity = torch.dot(target_vec, current_vec) / (torch.norm(target_vec) * torch.norm(current_vec))
            # (단어이름, 점수) 쌍으로 저장
            all_scores.append((target_word, similarity.item()))  
            # 점수가 높은 순서대로 정렬 (내림차순)
        all_scores.sort(key = lambda x : x[1], reverse = True)
            
        print(f"유추 결과 : [{a}] : [{b}] = [{c}] : ?")
            
        actual_top_k = min(top_k, len(all_scores))
        for i in range(actual_top_k):
            name, score = all_scores[i]
            print(f"AI가 추론한 정답: {name} (유사도: {score:.4f})")
                
                                
    
    
    find_most_similar("emma")
    solve_analogy("harry", "potter", "daniel")

# 테스트 실행 코드
# 1. Similarity Evaluation (단어 유사도 평가)
if __name__  == "__main__":
    run_test()