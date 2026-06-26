import models, constants
import converter.models as CVModels
import models

function_map = {
    "placeholder" : models.Value.from_placeholder,
    "call_function" : models.Operation.from_op,
    "output" : models.Output.from_output,
}

def simplify(messy_ir:CVModels.LayerMap):
    for value in messy_ir.nodes:
