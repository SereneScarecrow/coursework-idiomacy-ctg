import langchain

class ModelInference():
    def __init__(self, model, prompt):
        self.model = model
        self.prompt = prompt

    def __call__(self, *args, **kwds):
        pass