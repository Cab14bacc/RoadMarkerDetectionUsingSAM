import os
import sys

from ..configUtils.config import Config

class SplineTestConfig(Config):
    def get(self, field=None):
        if field not in self.config:
            print(f"Field '{field}' not found in config file: {self.config_path}")
            return None
        return self.config[field]
    
    def get_all_config(self):
        return self.config
    
    def get_parallel_dist_threshold(self):
        if self.config is not None:
            return self.get('SplineTest')['parallel_dist_threshold']
        return None
    
    def get_endpoint_dist_threshold(self):
        if self.config is not None:
            return self.get('SplineTest')['endpoint_dist_threshold']
        return None
    
    def get_pixel_cm(self):
        if (self.config is not None):
            return self.get('Common')['pixel_cm']
        else:
            return 1
    
    def get_variance_threshold(self):
        if (self.config is not None):
            return self.get('SplineTest')['variance_threshold']
        else:
            return 1

    def get_spline_gap_threshold(self):
        if (self.config is not None):
            return self.get('SplineTest')['spline_gap_threshold']
        else:
            return 1
        
    def get_spline_length_threshold(self):
        if (self.config is not None):
            return self.get('SplineTest')['spline_length_threshold']
        else:
            return 1
        

    def get_white_color(self):
        if self.config is not None:
            return self.get('SplineTest')['white_color']
        return None
    
    def get_yellow_color(self):
        if self.config is not None:
            return self.get('SplineTest')['yellow_color']
        return None
    
