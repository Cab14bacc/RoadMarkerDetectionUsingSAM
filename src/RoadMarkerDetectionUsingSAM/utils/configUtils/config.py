import yaml
import os

class Config:
    def __init__(self, config_path=None):
        self.config_path = config_path
        self.config = None
        if config_path is not None:
            self.config = self.load_config()

    def set_config_with_dict(self, config_dict):
        self.config = config_dict

    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def get(self):
        return self.config
  
