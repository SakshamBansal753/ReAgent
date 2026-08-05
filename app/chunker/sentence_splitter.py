from langchain_core.documents import Document
import spacy
nlp=spacy.load("en_core_web_sm")
class SentenceSplitter:
    """
    Splits langchain documents into sentence
    """
    def split(self,documents:list[Document]):
        sentence_docs=[]
        for doc in documents:
            parsed=nlp(doc.page_content)
            for sentence in parsed.sents:
                text=sentence.text.strip()
                if not text:
                    continue
                sentence_docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            doc.metadata.copy()
                        }
                    )
                )
        return sentence_docs        