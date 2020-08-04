import yaml
import os
config_path = os.path.join(os.path.dirname(__file__), "config.yml")
try:
    with open(config_path) as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError("Config file not found.")
    exit(1)
