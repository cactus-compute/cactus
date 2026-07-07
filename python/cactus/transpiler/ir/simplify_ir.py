"""
The code in this directory is responsible for taking in the raw components lists from the converter folder and transform
it into a valid computation DAG (consisting of both operation nodes and value nodes). Topological sort will then be performed
on the DAG to generate an in-order execution list.
"""

import models, constants
import converter.models as CVModels
import models

def simplify(messy_ir:CVModels.LayerMap):
    graph: models.Graph = models.Graph(messy_ir)

    

    
