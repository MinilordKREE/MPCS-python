import os
import time
import datetime
import pytz
import configparser
import regex
import pandas as pd
from itertools import islice
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
import json
import hashlib
import pickle
from typing import Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import math
import gc

OPENAI_API_KEY = "sk-proj-xxxxx"
ANTHROPIC_API_KEY = "sk-ant-api03-xxxxx"

# Cache directory for pickled responses
CACHE_DIR = "/net/scratch2/chenxi/LLM4MFD/all@once/cache"
PERPLEXITY_API_KEY = ""

class LlamaResponse:
    """
    A unified response object for local LLaMA or GPT-based calls.
    message.content -> model output text
    logprobs.content -> list of LogprobInfo objects (token-level info)
    """
    def __init__(self, message_content, logprobs_info):
        self.message = self.Message(message_content)
        self.logprobs = self.Logprobs(logprobs_info)

    class Message:
        def __init__(self, content):
            self.content = content

    class Logprobs:
        def __init__(self, content):
            self.content = content

    class LogprobInfo:
        def __init__(self, token, logprob, top_logprobs):
            self.token = token       # expected to be "0" or "1"
            self.logprob = logprob   # natural log probability (float)
            self.top_logprobs = top_logprobs  # list of tuples (token, logprob)

    class TopLogprob:
        def __init__(self, token, logprob):
            self.token = token
            self.logprob = logprob

class LLMManager:
    """
    Singleton that loads and caches the model+tokenizer.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
            cls._instance.current_model_name = None
        return cls._instance

    def load_model(self, model_name: str) -> None:
        """
        Load or reload the model if needed.
        """
        if self.model is not None and self.current_model_name == model_name:
            return  # Already loaded

        if self.model is not None:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            gc.collect()

        print(f"[DEBUG] Loading model: {model_name}")
        if "Deep" in model_name:
            repo_path = f"deepseek-ai/{model_name}"
        else:
            repo_path = f"meta-llama/{model_name}"

        self.model = AutoModelForCausalLM.from_pretrained(
            repo_path,
            torch_dtype=torch.float16,
            device_map="sequential",  # Let transformers handle multi-GPU placement
            max_memory={
                0: "76GiB",   # Tell HF that GPU 0 has only 20GiB available
                1: "76GiB",
                2: "76GiB",
                3: "76GiB",
            },
            offload_folder="/net/scratch2/chenxi/LLM4MFD/all@once/offload"  # Optional: offload extra weights to CPU
        )

        self.tokenizer = AutoTokenizer.from_pretrained(repo_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.current_model_name = model_name

    def generate(self, prompt, **generation_kwargs):
        """
        Generate text from the prompt and collect token-level logprobs.
        Debug prints are added to show the tokens and their computed probabilities.
        """
        chat = [
            {"role": "system", "content": "You are a helpful assistant that analyzes moral dimensions in text."},
            {"role": "user", "content": prompt}
        ]
        prompt_formatted = self.tokenizer.apply_chat_template(chat, tokenize=False)
        inputs = self.tokenizer(prompt_formatted,
                                return_tensors="pt",
                                truncation=True,
                                padding=True,
                                return_attention_mask=True)
        # IMPORTANT: Do not force inputs to a specific device.
        # Remove manual device placement so that the model's device_map (set to "auto")
        # will automatically route the inputs to the correct GPU(s).
        # device = next(iter(self.model.hf_device_map.values()))
        # inputs = {k: v.to(device) for k, v in inputs.items()}
        # self.model.to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs,
                return_dict_in_generate=True,
                output_scores=True
            )

        # Get tokens generated after the prompt
        gen_tokens = outputs.sequences[0][len(inputs["input_ids"][0]):]
        decoded_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        print(f"[DEBUG] Decoded text: {decoded_text}")

        # Debug: Print each token and its raw score (decoded)
        if hasattr(outputs, "scores") and outputs.scores:
            print("[DEBUG] Token-level details:")
            for token_id, scores_tensor in zip(gen_tokens, outputs.scores):
                token_str = self.tokenizer.decode([token_id])
                topk = torch.topk(scores_tensor[0], k=5)
                top_tokens = [self.tokenizer.decode([int(t)]) for t in topk.indices]
                top_scores = topk.values.tolist()
                print(f"Token: {token_str.strip()} (ID: {token_id}), Top tokens: {list(zip(top_tokens, top_scores))}")

        # Gather logprobs info
        logprobs_info = []
        if hasattr(outputs, "scores") and outputs.scores:
            zero_tokens, one_tokens = get_binary_token_ids(self.tokenizer)
            for token_id, scores_tensor in zip(gen_tokens, outputs.scores):
                result = process_binary_logprobs(
                    scores_tensor.unsqueeze(0),
                    token_id.item(),
                    zero_tokens,
                    one_tokens,
                    self.tokenizer
                )
                if result:
                    print(f"[DEBUG] Logprob info for token ID {token_id}: {result}")
                    logprobs_info.append(result)

        return decoded_text, logprobs_info

    def cleanup(self):
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.current_model_name = None
            torch.cuda.empty_cache()
            gc.collect()

def _cache_key(prompt: str) -> str:
    return hashlib.sha1(prompt.encode()).hexdigest()

def _cache_file_path(prompt: str, cache_dir: str, api_type: str, deployment_name: str) -> str:
    key = _cache_key(prompt)
    model_dir = os.path.join(cache_dir, api_type, deployment_name)
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"{key}.pkl")

def _load_from_cache(prompt: str, cache_dir: str, api_type: str, deployment_name: str) -> Optional[Any]:
    cfile = _cache_file_path(prompt, cache_dir, api_type, deployment_name)
    if os.path.exists(cfile):
        with open(cfile, "rb") as f:
            print(f"[DEBUG] Loading from cache: {cfile}")
            return pickle.load(f)
    return None

def _save_to_cache(prompt: str, response: Any, cache_dir: str, api_type: str, deployment_name: str) -> None:
    cfile = _cache_file_path(prompt, cache_dir, api_type, deployment_name)
    with open(cfile, "wb") as f:
        pickle.dump(response, f)
    print(f"[DEBUG] Saved response to cache: {cfile}")

def get_binary_token_ids(tokenizer):
    zero_list = tokenizer(" 0", add_special_tokens=False)['input_ids']
    zero_list += tokenizer("0", add_special_tokens=False)['input_ids']
    one_list  = tokenizer(" 1", add_special_tokens=False)['input_ids']
    one_list += tokenizer("1", add_special_tokens=False)['input_ids']
    return list(set(zero_list)), list(set(one_list))

def process_binary_logprobs(scores, curr_token_id, zero_tokens, one_tokens, tokenizer):
    """
    Compute the binary probability distribution over tokens "0" and "1".
    Handles extra dimensions by squeezing.
    """
    # Remove any extra dimensions (e.g., turn [1, 1, vocab_size] into [vocab_size])
    while scores.dim() > 1:
        scores = scores.squeeze(0)

    if scores.dim() != 1:
        print(f"[DEBUG] Unexpected shape after squeeze: {scores.shape}")
        return None

    logits_1d = scores  # Now shape should be (vocab_size,)
    
    # Compute softmax probabilities
    probs = torch.nn.functional.softmax(logits_1d, dim=-1)
    vocab_size = probs.size(0)
    print(f"[DEBUG] Vocab size: {vocab_size}")

    valid_zero = [z for z in zero_tokens if 0 <= z < vocab_size]
    valid_one = [o for o in one_tokens if 0 <= o < vocab_size]
    print(f"[DEBUG] Valid zero tokens: {valid_zero}")
    print(f"[DEBUG] Valid one tokens: {valid_one}")

    if not valid_zero and not valid_one:
        print("[DEBUG] No valid zero/one tokens found.")
        return None

    zprob = sum(probs[z].item() for z in valid_zero)
    oprob = sum(probs[o].item() for o in valid_one)
    print(f"[DEBUG] Raw zero_prob: {zprob}, one_prob: {oprob}")

    total = zprob + oprob
    if total <= 0:
        print("[DEBUG] Sum of probabilities is zero or negative.")
        return None

    zprob /= total
    oprob /= total
    print(f"[DEBUG] Normalized zero_prob: {zprob}, one_prob: {oprob}")

    chosen_token = None
    if curr_token_id in valid_zero:
        chosen_token = "0"
    elif curr_token_id in valid_one:
        chosen_token = "1"
    print(f"[DEBUG] Current token ID {curr_token_id} interpreted as: {chosen_token}")

    if chosen_token is not None:
        chosen_prob = zprob if chosen_token == "0" else oprob
        lp = math.log(chosen_prob) if chosen_prob > 0 else float("-inf")
        return {
            "token": chosen_token,
            "logprob": lp,
            "top_logprobs": [
                ("0", math.log(zprob) if zprob > 0 else float("-inf")),
                ("1", math.log(oprob) if oprob > 0 else float("-inf"))
            ]
        }
    return None

def api_call(prompt, deployment_name, temperature, max_tokens, top_p, logprobs, top_logprobs, overwrite=False):
    """
    Call an API or local model to get a LlamaResponse (text output and token-level logprobs).
    Uses caching (unless overwrite=True).
    """
    time.sleep(0.5)  # simple throttling
    api_key_prefix = deployment_name.split('-', 1)[0]
    cached = _load_from_cache(prompt, CACHE_DIR, api_key_prefix, deployment_name)
    if not overwrite and cached is not None:
        return cached

    if deployment_name.startswith("gpt-"):
        text = f"[OpenAI GPT placeholder for {deployment_name}]\nPrompt:\n{prompt}"
        llama_resp = LlamaResponse(text, [])
        _save_to_cache(prompt, llama_resp, CACHE_DIR, api_key_prefix, deployment_name)
        return llama_resp

    elif deployment_name.startswith("Llama") or deployment_name.startswith("Deep"):
        llm_manager = LLMManager()
        llm_manager.load_model(deployment_name)
        decoded_text, logprobs_info_list = llm_manager.generate(
            prompt,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=(temperature > 0),
            pad_token_id=llm_manager.tokenizer.pad_token_id,
        )
        resp = LlamaResponse(decoded_text, logprobs_info_list)
        _save_to_cache(prompt, resp, CACHE_DIR, api_key_prefix, deployment_name)
        return resp
    else:
        text = f"[Unrecognized deployment: {deployment_name}]\nPrompt:\n{prompt}"
        llama_resp = LlamaResponse(text, [])
        _save_to_cache(prompt, llama_resp, CACHE_DIR, api_key_prefix, deployment_name)
        return llama_resp
