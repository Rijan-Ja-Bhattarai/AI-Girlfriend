import tiktoken 
from typing import List


class Tokenizer:
    def __init__(self, vocab: str = "cl100k_base") -> None:
        """
        Constructor for AI Girlfriend 

        Sets the tokenizer for the model: uses the GPT 3.5/4 tokenizer 
        """
        self.tokenizer = tiktoken.get_encoding(vocab)

    def encode(self, text: str) -> List[int]:
        """
        Encoder function for AI girlfriend 

        Args:
            text (str) : The text to encode 
        
        Output:
            Encoded text (List[int]): Converts string into List of integers/ids to be used for training
        """
        return self.tokenizer.encode(text=text, allowed_special={'<|endoftext|>'})

    def decode(self, tokens: List[int]) -> str:
        """
        Decoder function for AI girlfriend 

        Args:
            Tokens (List[int]): Encoded text 
        
        Output:
            Decoded tokens (str): String of text which was decoded from token ids
        """
        return self.tokenizer.decode(tokens=tokens, errors='strict') 

    def vocab_size(self) -> int:
        """
        Returns vocabulary size being used by current tokenizer
        """
        return self.tokenizer.n_vocab