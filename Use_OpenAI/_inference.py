import os
from dotenv import load_dotenv

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)   # for exponential backoff

# Load OPENAI API Key
load_dotenv()

client = OpenAI(
    api_key = os.getenv('OPENAI_API_KEY')
)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def prompt_openai(question: str,
                  openai_model='gpt-4o-mini') -> str:
    completion = client.chat.completions.create(
        model=openai_model,
        messages=[
            {'role': 'system', 'content': 'You are a helpful system'},
            {'role': 'user', 'content': f'{question}'},
        ]
    )

    return str(completion.choices[0].message.content)