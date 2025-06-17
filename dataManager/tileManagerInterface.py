import numpy as np
import cv2
from .tileData import TileData
import os

'''
tile manager to split image and data size to each tile as {tile_size} pixel. 
If smaller than 1024, remain unchanged

init argument:
    source_image: original size image read from cv2
    input_points_list: list of point set
    input_labels_list: list of label set
    tile_size: split tile to each size
'''
class TileManagerInterface:
    def split_tile_position(self, h, w):
        (x_coords, y_coords), (x_covers, y_covers) = self.compute_tile_position((h, w))
        return (x_coords, y_coords), (x_covers, y_covers)

    def compute_tile_position(self, shape):
        h, w = shape
        tile_size = self.tile_size

        def compute_positions(length):
            positions = list(range(0, max(length - tile_size + 1, 1), tile_size))
            if (length > tile_size):
                last_start = length - tile_size
                if last_start not in positions:
                    positions.append(last_start)

            return positions
        
        coords_y = compute_positions(h)
        coords_x = compute_positions(w)

        offset = tile_size // 2

        offset_x = [x + offset for x in coords_x if x + offset + tile_size <= w]
        offset_y = [y + offset for y in coords_y if y + offset + tile_size <= h]


        return (coords_x, coords_y), (offset_x, offset_y)
