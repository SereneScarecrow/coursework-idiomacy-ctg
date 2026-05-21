from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler

from model_constructor import ModelConstructor
from model_inference import PromptConstructor, ModelInference

load_dotenv()

client = ModelConstructor.create_client("GigaChat-2", "gigachat")
prompt = PromptConstructor.get_prompt("test", {"word": 'hello'})

print(prompt)

inference = ModelInference(client)
print(inference(prompt))

