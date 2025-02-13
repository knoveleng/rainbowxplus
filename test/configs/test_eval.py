import ast
from rainbowxplus.configs.eval import EvalConfigLoader


def test_read_config():
    config = EvalConfigLoader.load("./configs/eval.yml")
    print(config)


test_read_config()
