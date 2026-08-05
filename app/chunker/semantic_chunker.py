from langchain_core.documents import Document
from .token_counter import TokenCounter
from .sentence_embedding import SentenceEmbedding
from .similarity import SimilarityCalculator
class SemanticChunker:
    def __init__(self,max_tokens:int=512,overlap_tokens:int=50,similarity_threshold:float=0.75,model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.max_tokens=max_tokens
        self.overlap_tokens=overlap_tokens
        self.token_counter=TokenCounter(model_name=model_name)
        self.similarity_threshold = similarity_threshold
        self.embedder = SentenceEmbedding(model_name)
        self.similarity = SimilarityCalculator()

    def chunker(self, sentence_docs: list[Document]) -> list[Document]:
        if not sentence_docs:
            return []

        sentences = [doc.page_content for doc in sentence_docs]
        embeddings = self.embedder.encode(sentences)
        chunks = []
        current_docs = []
        current_tokens = 0

        def flush():
            nonlocal current_docs, current_tokens
            if not current_docs:
                return
            chunk_text = " ".join(doc.page_content for doc in current_docs)
            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata=current_docs[0].metadata.copy(),
                )
            )

        for i, sentence_doc in enumerate(sentence_docs):
            sentence = sentence_doc.page_content
            token_count = self.token_counter.count_tokens(sentence)
            similarity = 1.0 if i == 0 else self.similarity.cosine(
                embeddings[i - 1], embeddings[i]
            )

            # Oversized sentence: flush what we have, emit it as its own chunk
            if token_count > self.max_tokens:
                flush()
                chunks.append(
                    Document(page_content=sentence, metadata=sentence_doc.metadata.copy())
                )
                current_docs = []
                current_tokens = 0
                continue

            if current_tokens + token_count > self.max_tokens or similarity < self.similarity_threshold:
                flush()

                overlap = []
                overlap_count = 0
                for sent in reversed(current_docs):
                    t = self.token_counter.count_tokens(sent.page_content)
                    if overlap_count + t > self.overlap_tokens:
                        break
                    overlap.insert(0, sent)
                    overlap_count += t

                current_docs = overlap
                current_tokens = overlap_count

            current_docs.append(sentence_doc)
            current_tokens += token_count

        flush()
        return chunks
        