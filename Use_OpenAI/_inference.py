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
    """
    Prompting OpenAI in zero-shot setting

    Arguments
    : question: The question you will ask to the model
    : openAI_model: Model you will be using. Full list of models are available here: https://platform.openai.com/docs/models
    """
    completion = client.chat.completions.create(
        model=openai_model,
        messages=[
            {'role': 'system', 'content': "OPTIONAL: Place in a sample system prompt."},                    # e.g., You are a excited shopping webpage rater
            {'role': 'user', 'content': f"{question}"},
        ]
    )

    return str(completion.choices[0].message.content)

# 1-shot
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def prompt_openai(question: str,
                  openai_model='gpt-4o-mini') -> str:
    """
    Prompting OpenAI in n-shot setting.
    If you will like to increase the value of n, create more combinations of 'user', 'assistant' pairs

    Arguments
    : question: The question you will ask to the model
    : openAI_model: Model you will be using. Full list of models are available here: https://platform.openai.com/docs/models
    """

    completion = client.chat.completions.create(
        model=openai_model,
        messages=[
            {'role': 'system', 'content': "OPTIONAL: Place in a sample system prompt."},                    # e.g., You are a excited shopping webpage rater
            {'role': 'user', 'content': "Replace this with some question you will ask."},                   # e.g., ate the product 'Snoopy Swing Ornaments' on a scale of 1-5 and give me a reasoning.
            {'role': 'assistant', 'content': "Replace this with a sample format of answer you will like."}, # e.g., 5 because it is cute
            {'role': 'user', 'content': f'{question}'},
        ]
    )

    return str(completion.choices[0].message.content)