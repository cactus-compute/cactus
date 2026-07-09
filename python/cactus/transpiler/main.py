import converter.convert as cv
import converter.models as CVModels

#TODO: Update converter to also downlaod model config.txt
if __name__ == "__main__":
    model_map:CVModels = cv.model_ops_to_json("LiquidAI/LFM2.5-8B-A1B", "/Users/sandhup/Documents/personal/cactus/python/cactus/transpiler/converter/jsons/")
    