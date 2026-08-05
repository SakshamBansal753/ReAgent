from transformers import AutoTokenizer

class TokenCounter:
    def __init__(
        self,
        model_name:str="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
    def count_tokens(self,text:str)->int:
        """"
        Count the number of token in a text using tokenizer"""
        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False
            )
        )    