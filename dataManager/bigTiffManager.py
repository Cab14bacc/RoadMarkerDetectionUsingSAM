from .tileManagerInterface import TileManagerInterface
from .tileData import TileData

import tifffile
import cv2
from color_filter import ColorFilter

class BigTiffManager(TileManagerInterface):
    def __init__(self, input_path, output_path,  tile_size=1024):
        self.source_image_path = input_path
        self.output_path = output_path
        self.tile_size = tile_size

        self.image_map = tifffile.memmap(self.source_image_path)

        self.split_tile_list = []

    def get_split_tile_coords(self):
        height, width, channels = self.image_map.shape
        (x_coords, y_coords), (x_covers, y_covers) = self.split_tile_position(height, width)
        tile_size = self.tile_size

        x_coord_list, y_coord_list = [], []
        for y_start in y_coords:
            for x_start in x_coords:
                x_coord_list.append(x_start)
                y_coord_list.append(y_start)
        return x_coord_list, y_coord_list

    def get_image_tile(self, y_start, x_start):
        tile_size = self.tile_size
        x_end = min(x_start + tile_size, self.image_map.shape[1])
        y_end = min(y_start + tile_size, self.image_map.shape[0])
        tile = self.image_map[y_start:y_end, x_start:x_end]
        tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        return tile

    def get_list_of_tiles(self):
        return self.split_tile_list

    def get_image_from_start_point(self, start_coord):
        tile_size = self.tile_size
        x_start, y_start = start_coord
        x_end = min(x_start + tile_size, self.image_map.shape[1])
        y_end = min(y_start + tile_size, self.image_map.shape[0])
        
        tile = self.image_map[y_start:y_end, x_start:x_end]
        tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        return tile
