from typing import Dict, Any
import yaml


def load_yaml_safe(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, "r") as stream:
        try:
            content = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to load yaml from file: {yaml_path}") from e
    return content
