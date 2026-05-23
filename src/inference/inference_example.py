from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler

from inference.model_constructor import ModelConstructor
from inference.model_inference import PromptConstructor, ModelInference

load_dotenv()

client = ModelConstructor.create_client("GigaChat-2", "gigachat")
inference = ModelInference(client)

prompt_builder = PromptConstructor("test")
formated_prompt = prompt_builder.render({"word": 'hello'})

print(formated_prompt)
print(inference(formated_prompt))

