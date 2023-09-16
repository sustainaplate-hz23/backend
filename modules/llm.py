import os
import json
import requests
from dotenv import load_dotenv
from typing import Any
from llama_index.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.llms.base import llm_completion_callback


load_dotenv()


class BamLLM(CustomLLM):
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=int(os.getenv('CONTEXT_WINDOW')),
            num_output=int(os.getenv('MAX_OUTPUT_TOKENS')),
            model_name=os.getenv('MODEL_NAME')
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        data = {
            "model_id": os.getenv('MODEL_NAME'),
            "inputs": [prompt],
            "parameters": {
                "temperature": float(os.getenv('TEMPERATURE')),
                "max_new_tokens": int(os.getenv('MAX_OUTPUT_TOKENS'))
            }
        }
        headers = {
            "Authorization": f"Bearer {os.getenv('GENAI_KEY')}",
        }
        response = requests.post(os.getenv('GENAI_API'), json=data, headers=headers)

        if response.status_code == 200:
            result = response.json().get('results', [])
            if len(result) > 0:
                generated_text = result[0].get('generated_text')
                return CompletionResponse(text=generated_text)
        else:
            print(f"Error: {response.status_code} - {response.text}")
        raise NotImplementedError()

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        data = {
            "model_id": os.getenv('MODEL_NAME'),
            "inputs": [prompt],
            "parameters": {
                "temperature": float(os.getenv('TEMPERATURE')),
                "max_new_tokens": int(os.getenv('MAX_OUTPUT_TOKENS')),
                "stream": True
            }
        }
        headers = {
            "Authorization": f"Bearer {os.getenv('GENAI_KEY')}",
        }
        response = requests.post(os.getenv('GENAI_API'), json=data, headers=headers, stream=True)

        def gen():
            content = ""
            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=4096):
                    try:
                        if chunk:
                            output_str = chunk.decode('utf-8')
                            if output_str.startswith('data: '):
                                output_str = output_str[len('data: '):]
                            data_ = json.loads(output_str)
                            generated_text = data_['results'][0]['generated_text']
                            yield CompletionResponse(text=content, delta=generated_text)
                    except Exception as ex:
                        print(str(ex))
            else:
                yield CompletionResponse(text="Network Error")

        return gen()