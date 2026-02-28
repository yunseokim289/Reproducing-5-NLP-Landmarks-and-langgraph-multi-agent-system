import torch
import matplotlib
# ✅ 화면 충돌 방지를 위해 반드시 맨 윗줄에 배치
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.dataset import Word2VecDataset

def visualize_results():
    print("🚀 시각화 프로세스를 시작합니다...")

    # 1. 데이터 및 모델 로드
    try:
        dataset = Word2VecDataset("data/toy_data.txt")
        checkpoint = torch.load("word2vect_model.pth", map_location="cpu")
        embeddings = checkpoint['in_embed.weight'].numpy()
        print("✅ 모델 가중치 로드 완료.")
    except Exception as e:
        print(f"❌ 로드 중 에러 발생: {e}")
        return

    # 2. 시각화할 핵심 단어 리스트 (인턴님이 테스트한 단어 중심)
    words_to_visualize = [
        "harry", "potter", "ron", "hermione", 
        "emma", "watson", "rupert", "grint", "daniel", "radcliffe",
        "hogwarts", "magic", "movie", "director", "dumbledore", "voldemort"
    ]
    
    # 단어장에 존재하는 단어만 선별
    valid_words = [w for w in words_to_visualize if w in dataset.word2idx]
    indices = [dataset.word2idx[w] for w in valid_words]
    selected_vectors = embeddings[indices]

    # 3. PCA를 이용한 차원 축소 (100D -> 2D)
    pca = PCA(n_components=2)
    result = pca.fit_transform(selected_vectors)

    # 4. 그래프 그리기
    plt.figure(figsize=(12, 10))
    plt.scatter(result[:, 0], result[:, 1], edgecolors='k', c='skyblue', s=100)

    # 점 옆에 단어 텍스트 표시
    for i, word in enumerate(valid_words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]), xytext=(5, 5),
                     textcoords='offset points', fontsize=11, fontweight='bold')

    plt.title("Step 6: Word2Vec Embedding Visualization (PCA)", fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 5. 파일 저장
    output_path = "word_embedding_plot.png"
    plt.savefig(output_path, dpi=300)
    print(f"✨ 시각화 완료! '{output_path}' 파일이 생성되었습니다.")

if __name__ == "__main__":
    visualize_results()