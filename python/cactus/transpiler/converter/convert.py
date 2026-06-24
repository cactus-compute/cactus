import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

from model_preprocessing import load_model, load_input



def model_ops_to_json(id:str, output_path:str):
    model = load_model(id)
    input = load_input()

    model.config.return_dict = False if hasattr(model, "config") else None 
    model.config.use_cache = False if hasattr(model, "use_cache") else None

    with torch.no_grad():
        exported_model = torch.export.export(model, args=(), kwargs=dict(input), strict=False).run_decompositions()

    #Iterates across every node in the graph and stores it as a layer object
    for i, record in enumerate(exported_model.graph_module.graph.nodes):
        







if __name__ == "main":
    load_model(input("Enter model ID: "))


