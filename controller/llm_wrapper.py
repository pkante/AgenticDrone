import os
from pathlib import Path

import openai
from openai import Stream, ChatCompletion

GPT3 = "gpt-3.5-turbo-16k"
GPT4 = "gpt-4"
GPT5 = "gpt-5"
LLAMA3 = "meta-llama/Meta-Llama-3-8B-Instruct"

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
chat_log_path = CURRENT_DIR / "assets" / "chat_log.txt"


def _load_env_file():
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key, value)


_load_env_file()

class LLMWrapper:
    def __init__(self, temperature=0.0):
        self.temperature = temperature
        self.llama_client = openai.OpenAI(
            # base_url="http://10.66.41.78:8000/v1",
            base_url="http://localhost:8000/v1",
            api_key="token-abc123",
        )
        self.gpt_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def request(self, prompt, model_name=GPT5, stream=False) -> str | Stream[ChatCompletion.ChatCompletionChunk]:
        if model_name == LLAMA3:
            client = self.llama_client
        else:
            client = self.gpt_client
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stream=stream,
        )

        # save the message in a txt
        with chat_log_path.open("a", encoding="utf-8") as f:
            f.write(prompt + "\n---\n")
            if not stream:
                f.write(response.model_dump_json(indent=2) + "\n---\n")

        if stream:
            return response

        return response.choices[0].message.content
