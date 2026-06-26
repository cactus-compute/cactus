import converter.convert as cv
import converter.models as CVModels

if __name__ == "__main__":
    model_map:CVModels = cv.model_ops_to_json("google/gemma-4-E2B", "/Users/sandhup/Documents/personal/cactus/python/cactus/transpiler/converter/jsons/")
    