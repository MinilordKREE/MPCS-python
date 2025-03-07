from openai import OpenAI
import os
import time
import datetime
import pytz
import configparser
import regex
import pandas as pd
# from anytree import Node, RenderTree
# import tiktoken
from itertools import islice
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI
# import anthropic
import json
import hashlib
import pickle
from typing import Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
# from llama_cpp import Llama
from sklearn import metrics

OPENAI_API_KEY = ""

def _cache_key(prompt: str) -> str:
    return hashlib.sha1(prompt.encode()).hexdigest()

def _cache_file_path(prompt: str, cache_dir:str, api_type: str, deployment_name:str) -> str:
    cache_key = _cache_key(prompt)
    api_dir = os.path.join(cache_dir, api_type)
    model_dir = os.path.join(api_dir, deployment_name)
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"{cache_key}.pkl")

def _load_from_cache(prompt: str, cache_dir:str, api_type: str, deployment_name:str) -> Optional[Any]:
    cache_file = _cache_file_path(prompt, cache_dir, api_type, deployment_name)
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None

def _save_to_cache(prompt: str, response: Any, cache_dir:str, api_type: str, deployment_name:str) -> None:
    cache_file = _cache_file_path(prompt, cache_dir, api_type, deployment_name)
    with open(cache_file, "wb") as f:
        pickle.dump(response, f)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20))
def api_call(prompt, deployment_name, temperature, max_tokens, top_p, logprobs, top_logprobs, overwrite = False):
    """
    Call API (OpenAI, Azure, Perplexity) and return response
    - prompt: prompt template
    - deployment_name: name of the deployment to use (e.g. gpt-4, gpt-3.5-turbo, etc.)
    - temperature: temperature parameter
    - max_tokens: max tokens parameter
    - top_p: top p parameter
    """
    time.sleep(0.5)  # Change to avoid rate limit

    if deployment_name in ["gpt-35-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "DeepSeek-R1-Distill-Llama-8B","deepseek-chat"]:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=deployment_name,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            top_p=float(top_p),
            logprobs = logprobs,
            top_logprobs = top_logprobs,
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
        )
        _save_to_cache(prompt, response.choices[0],
                       CACHE_DIR, deployment_name.split('-', 1)[0], deployment_name)
        return response.choices[0]