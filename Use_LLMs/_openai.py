import os

from typing import List, Dict

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)   # for exponential backoff

# from dotenv import load_dotenv
# # Load OPENAI API Key
# load_dotenv()

class BasicOpenAI:
    def __init__(self,
                 model:str = 'gpt-4o-mini',
                 api_key:str = None) -> None:
        
        self.model = model
        self.client = OpenAI(api_key = api_key)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def zero_shot(self,
                  sys_prompt:str=None,
                  question:str=None,) -> str:
        """
        Prompting OpenAI in zero-shot setting

        Arguments
        : question: The question you will ask to the model
        """

        chat_messages = []

        if sys_prompt:
            chat_messages.append({'role': 'system', 'content': f"{sys_prompt}"})

        if not question:
            # No input question.
            return 'DNE'

        completion = self.client.chat.completions.create(
            model = self.model,
            messages=[
                {'role': 'system', 'content': ""},
                {'role': 'user', 'content': f"{question}"}
            ]
        )

        return str(completion.choices[0].message.content)