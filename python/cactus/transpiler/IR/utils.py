from . import models, constants
from ..Fusions import models as FModels
from ..Fusions import fusions as f
from ..Converter import models as CModels


def match_nodes(synth_node: models.Node, graph: models.Graph, ) -> bool:
