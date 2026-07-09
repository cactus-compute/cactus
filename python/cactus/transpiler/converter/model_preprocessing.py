from . import constants
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(id:str) -> object:
    model = AutoModelForCausalLM.from_pretrained(id, attn_implementation = "eager")
    model.eval()
    return model

#Currently just handles single input type (strings), but will change to make input a list to allow mutli-modal models to run multiple different inputs
def load_input(id:str) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(id)
    input = (tokenizer(constants.DEFAULT_PROMPT, return_tensors="pt")["input_ids"],)
    return input
