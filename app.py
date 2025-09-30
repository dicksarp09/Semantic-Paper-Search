import gradio as gr
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1️⃣ Load your dataset
papers_path = r"C:\Users\Dickson\SEMANTIC SEARCH\Data\processed\arxiv_clean.csv"
papers = pd.read_csv(papers_path)

# Ensure 'text' column exists
if 'text' not in papers.columns:
    papers['text'] = papers['title'] + " " + papers['summary']

# 2️⃣ Load your fine-tuned model
model_path = r"C:\Users\Dickson\SEMANTIC SEARCH\notebooks\output\fine_tuned_papers_light"
model = SentenceTransformer(model_path)

# 3️⃣ Generate embeddings and build FAISS index
embeddings = model.encode(papers['text'].tolist(), show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# 4️⃣ Define a search function
def search_papers(query, top_k=5):
    query_emb = model.encode([query]).astype("float32")
    scores, indices = index.search(query_emb, top_k)
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        score = scores[0][i]
        title = papers.iloc[idx]['title']
        summary = papers.iloc[idx]['summary']
        results.append(f"{title} (Score: {score:.4f})\n{summary}\n")
    return "\n".join(results)

# 5️⃣ Build Gradio UI
iface = gr.Interface(
    fn=search_papers,
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here..."),
    outputs=gr.Textbox(lines=15, placeholder="Top results will appear here..."),
    title="Semantic Paper Search",
    description="Search arXiv papers using your fine-tuned NLP model!"
)

# 6️⃣ Launch app
iface.launch(share=True)
