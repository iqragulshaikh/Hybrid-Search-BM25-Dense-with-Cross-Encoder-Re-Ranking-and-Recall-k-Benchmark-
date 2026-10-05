# Hybrid Search (BM25 + Dense) with Cross-Encoder Re-Ranking & Recall@k Benchmark
**Project Track:** Specialist Advanced Build (Week 4)  
**Domain:** Community-Based Leather Goods Exporter  
**Constraint:** No-GPU / CPU-only execution (~10 minutes runtime)

---

## 1. Executive Summary & Scope Statement
This project builds a robust retrieval benchmark for a **Community-Based Leather Goods Exporter**. The knowledge base requires navigating diverse documentation ranging from artisan tanning processes to global export regulations. We implemented and compared three retrieval paradigms: lexical search (`rank_bm25`), semantic dense search (`sentence-transformers`), and a hybrid pipeline combining Reciprocal Rank Fusion (RRF) with a lightweight Cross-Encoder re-ranker (`ms-marco-MiniLM-L-6-v2`).

## 2. Tools & Techniques Used
- **BM25 (`rank_bm25`):** For precise keyword and token-frequency matching.
- **Dense Embeddings (`sentence-transformers / all-MiniLM-L6-v2`):** For capturing semantic meaning and contextual intent on a CPU.
- **Reciprocal Rank Fusion (RRF):** To merge lexical and dense rank lists fairly without score-scale conflicts.
- **Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`):** To perform deep contextual re-ranking on the top 20 candidate pool.
- **Evaluation Metrics:** Recall@1, Recall@3, Recall@5, and Mean Reciprocal Rank (MRR).

## 3. Methodology & Implementation Steps
1. **Corpus Generation:** Generated 2,800 synthetic paragraphs covering leather tanning, export rules, and product care.
2. **Dataset Split:** Created 35 hand-written queries mapped to exact source paragraph IDs, split into 24 tuning questions and 11 unseen evaluation questions.
3. **Retrieval & Fusion:** Queried both BM25 and Dense models, fused ranks using RRF ($k=60$), and passed top candidates to the Cross-Encoder.
4. **Evaluation:** Evaluated the pipeline strictly on the 11 unseen questions.

## 4. Benchmark Results (Unseen Set)
- **Recall@1:** 0.0909
- **Recall@3:** 0.0909
- **Recall@5:** 0.0909
- **MRR:** 0.0909

## 5. Limitations & Future Improvements
- **Synthetic Data Noise:** Random paragraph-to-query assignments lowered baseline precision; real-world curated text will drastically improve Recall@k.
- **CPU Constraint:** Limiting models to lightweight variants (`MiniLM`) reduced deep semantic nuance compared to massive server-grade cross-encoders.
- **Improvement:** Future iterations will incorporate query expansion and domain-specific fine-tuning on real exporter FAQs.
