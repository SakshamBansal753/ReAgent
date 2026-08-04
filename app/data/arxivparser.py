from pathlib import Path
import arxiv
import fitz
from langchain_core.documents import Document

class ArxivParser:
    """
    A class to parse arxiv papers and extract their content in the form of Langchian Document objects.
    """
    def __init__(self,download_dir:str="downloads"):
        self.download_dir=Path(download_dir)
        self.download_dir.mkdir(parents=True,exist_ok=True)

    def search_papers(self,query:str,max_results:int=2):
        """"
        search for papers using arxiv for a user query and return a list of arxiv results
        """
        search=arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        return list(search.results())

    def download_pdf(self,paper)->str:
        """
        Download paper pdf
        
        """
        pdf_path=self.download_pdf(dirpath=str(self.download_dir))
        return pdf_path


    def parse_pdf(self,pdf_path:str,metadata:dict)->list[Document]:
        """"
        Extract text from pdf
        """
        pdf=fitz.open(pdf_path)
        documents=[]
        for page_number,page in enumerate(pdf):
            text=page.get_text()
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        **metadata,
                        "page_no":page_number+1
                    }
                )
            )
        pdf.close()
        return documents

    def load(self,query:str,max_results:int=2)->list[Document]:
        """
        search-> download->Parse papers
        """
        papers=self.search_papers(query,max_results)
        all_documents=[]
        for paper in papers:
            try:
                pdf_path=self.download_pdf(paper)
                metadata={
                  "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "published": str(paper.published),
                    "summary": paper.summary,
                    "entry_id": paper.entry_id,
                    "source": pdf_path,
                }

                docs = self.parse_pdf(pdf_path, metadata)

                all_documents.extend(docs)

            except Exception as e:
                print(f"Error processing {paper.title}: {e}")

        return all_documents
