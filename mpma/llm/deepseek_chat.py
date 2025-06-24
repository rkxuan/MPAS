import asyncio
import os
from dataclasses import asdict
from typing import List, Union, Optional
from dotenv import load_dotenv
import random
import async_timeout
from openai import OpenAI, AsyncOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import time
from typing import Dict, Any
import requests
import aiohttp

from mpma.llm.format import LLM_Message
from mpma.llm.price import cost_count
from mpma.llm.llm import LLM
from mpma.llm.llm_registry import LLMRegistry

load_dotenv()
OPENAI_API_KEYS=[os.getenv(f"OPENAI_API_KEY")]
for i in range(10):
    if os.getenv(f"OPENAI_API_KEY{i}"):
        OPENAI_API_KEYS.append(os.getenv(f"OPENAI_API_KEY{i}"))
BASE_URL = os.getenv(f"BASE_URL")

def deepseek_chat(
    model: str,
    messages: List[LLM_Message],
    max_tokens: int = 8192,
    temperature: float = 0.0,
    num_comps=1,
    return_cost=False,
) -> Union[List[str], str]:
    if messages[0].content == '$skip$':
        return ''

    api_key = random.sample(OPENAI_API_KEYS, 1)[0]
    headers = {'Authorization': 'Bearer '+api_key, 
               'Content-Type': 'application/json'}
    data = {
        'model': 'DeepSeek-V3',
        'messages': [asdict(message) for message in messages],
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_p': 1,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0,
    }

    response = requests.post(BASE_URL, headers=headers, json=data)

    return response.json()['choices'][0]['message']['content']


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(10))
async def deepseek_achat(
    model: str,
    messages: List[LLM_Message],
    max_tokens: int = 8192,
    temperature: float = 0.0,
    num_comps=1,
    return_cost=False,
) -> Union[List[str], str]:
    if messages[0].content == '$skip$':
        return '' 

    api_key = random.sample(OPENAI_API_KEYS, 1)[0]
    headers = {'Authorization': 'Bearer '+api_key, 
               'Content-Type': 'application/json'}
    data = {
        'model': 'DeepSeek-V3',
        'messages': [asdict(message) for message in messages],
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_p': 1,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0,
    }

    try:
        async with async_timeout.timeout(1000):
            session = aiohttp.ClientSession()
            result = await session.post(BASE_URL, json=data, headers=headers)
            response = await result.json()
    except asyncio.TimeoutError:
        print('Timeout')
        raise TimeoutError("DeepSeek Timeout")
    finally:
        await session.close()
    return response['choices'][0]['message']['content']

@LLMRegistry.register('DeepSeekChat')
class DeepSeekChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[LLM_Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [LLM_Message(role="user", content=messages)]

        return await deepseek_achat(self.model_name,
                               messages,
                               max_tokens,
                               temperature,
                               num_comps)

    def gen(
        self,
        messages: List[LLM_Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [LLM_Message(role="user", content=messages)]

        return deepseek_chat(self.model_name,
                        messages, 
                        max_tokens,
                        temperature,
                        num_comps)
