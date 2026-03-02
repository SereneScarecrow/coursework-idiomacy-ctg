import os
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
from model_inference import ModelConstructor, PromptConstructor, ModelInference

load_dotenv()

langfuse_handler = CallbackHandler()


model = ModelConstructor.create_client("meta-llama/llama-3.3-70b-instruct:free", "openrouter")
prompt = PromptConstructor.get_prompt("test")
inference = ModelInference(model, langfuse_handler)
print(inference(prompt))

