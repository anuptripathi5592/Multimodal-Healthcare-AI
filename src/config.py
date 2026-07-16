import yaml
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        return self.config[key]
