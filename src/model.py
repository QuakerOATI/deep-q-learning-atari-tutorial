from keras import layers, Model
from pathlib import Path
import yaml

config_file = Path(__file__).with_suffix(".yml")


def parse_config(path):
    with open(path) as file:
        config = yaml.safe_load(file)
    return config


def create_qnetwork(config):
    prev = None
    inputs = None
    for layer in config["layers"]:
        curr = layers.__dict__.get(layer.get("type", "Conv2D"))
        curr = curr(*layer.get("args", []), **layer.get("kwargs", {}))
        if prev is not None:
            curr = curr(prev)
        if inputs is None:
            inputs = curr
        prev = curr
    return Model(inputs=inputs, outputs=prev)
