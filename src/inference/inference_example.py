import os
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
from model_inference import ModelConstructor, PromptConstructor, ModelInference

load_dotenv()

model = ModelConstructor.create_client("llama3.2:3b", "ollama")
prompt = PromptConstructor.get_prompt("test", {"word": 'hello'})
print(prompt)
inference = ModelInference(model)
print(inference(prompt))

