import os
from dotenv import load_dotenv

from typing import List, Dict

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)   # for exponential backoff

# Load OPENAI API Key
load_dotenv()

class BasicOpenAI:
    def __init__(self,
                 model:str = 'gpt-4o-mini') -> None:
        
        self.model = model
        self.client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def zero_shot(self,
                  question:str,) -> str:
        """
        Prompting OpenAI in zero-shot setting

        Arguments
        : question: The question you will ask to the model
        """

        completion = self.client.chat.completions.create(
            model = self.model,
            messages=[
                {'role': 'system', 'content': ""},
                {'role': 'user', 'content': f"{question}"}
            ]
        )

        return str(completion.choices[0].message.content)