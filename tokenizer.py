import tiktoken 
from typing import List


class Tokenizer:
    def __init__(self) -> None:
        """
        Constructor for AI Girlfriend 

        Sets the tokenizer for the model: uses the GPT 2 tokenizer 
        """
        self.enc = tiktoken.get_encoding("gpt-2")

    def encode(self, text: str) -> List[int]:
        """
        Encoder function for AI girlfriend 

        Args:
            text (str) : The text to encode 
        
        Output:
            Encoded text (List[int]): Converts string into List of integers/ids to be used for training
        """
        return self.enc.encode(text=text, allowed_special='{<|endoftext|>}')

    def decode(self, tokens: List[int]) -> str:
        """
        Decoder function for AI girlfriend 

        Args:
            Tokens (List[int]): Encoded text 
        
        Output:
            Decoded tokens (str): String of text which was decoded from token ids
        """
        return self.enc.decode(tokens=tokens, errors='strict') 