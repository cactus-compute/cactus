import torch
import json
from models import LayerRecord, LayerMap
from model_preprocessing import load_model, load_input
import constants



def model_ops_to_json(id:str, output_path:str):
    model = load_model(id)
    input = load_input(id)

    model.config.return_dict = False if hasattr(model.config, "return_dict") else None 
    model.config.use_cache = False if hasattr(model.config, "use_cache") else None

    with torch.no_grad():
        exported_model = torch.export.export(model, args=input, kwargs={"use_cache":False}, strict=False).run_decompositions()

    #Iterates across every node in the graph and stores it as a layer object
    nodes = []
    for i, record in enumerate(exported_model.graph_module.graph.nodes):
        nodes.append(LayerRecord.from_node(num = i, x = record))
    
    json_ = LayerMap.from_data(x=exported_model, name=id, model_task = constants.MODEL_TASK, nodes_list=nodes).model_dump_json(indent=4)
    
    with open(output_path+"output.json", "w", encoding="utf-8") as file:
        file.write(json_)



if __name__ == "__main__":
    model_ops_to_json("google/gemma-4-E2B", "/Users/sandhup/Documents/personal/cactus/python/cactus/transpiler/converter/jsons/")
    print("Done")

