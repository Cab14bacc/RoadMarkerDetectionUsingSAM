import numpy as np

class TileData:
    def __init__(self, image, start_coord, input_points_list, input_labels_list):
        self.image = image
        self.start_coord = start_coord
        self.input_points_list = input_points_list
        self.input_labels_list = input_labels_list
    
    def get_image(self):
        return self.image
    
    def get_start_coord(self):
        return self.start_coord
    
    def get_input_list(self):
        return self.input_points_list, self.input_labels_list