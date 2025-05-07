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
class TileManager:
    def __init__(self, source_image, input_points_list, input_labels_list, mask_path, tile_size=1024):

        self.source_tile = TileData(
            image=source_image,
            start_coord=[0, 0],
            input_points_list=input_points_list,
            input_labels_list=input_labels_list,
        )
        self.tile_size = tile_size
        self.split_tile_list = []

    def split_tile(self):
        h, w, _ = self.source_tile.get_image().shape
        (x_coords, y_coords), (x_covers, y_covers) = self._compute_tile_position((h, w))

        self.split_tile_list = []

        self._collect_tiles(x_coords, y_coords)
        self._collect_tiles(x_covers, y_covers)

    def _collect_tiles(self, x_starts, y_starts):
        image = self.source_tile.get_image()
        input_points_list, input_labels_list = self.source_tile.get_input_list()
        tile_size = self.tile_size

        for y_start in y_starts:
            for x_start in x_starts:
                x_end = x_start + tile_size
                y_end = y_start + tile_size

                tile = image[y_start:y_end, x_start:x_end].copy()
                local_points_list, local_labels_list = self._split_input_list_from_range(input_points_list, input_labels_list, [x_start, x_end], [y_start, y_end])

                if (local_points_list):
                    tile_data = TileData(
                        image=tile,
                        start_coord=[x_start, y_start],
                        input_points_list=local_points_list,
                        input_labels_list=local_labels_list,
                    )
                    self.split_tile_list.append(tile_data)


    def _compute_tile_position(self, shape):
        h, w = shape
        tile_size = self.tile_size

        def compute_positions(length):
            positions = list(range(0, length, tile_size))
            if length > tile_size and length % tile_size != 0:
                if (length - tile_size) not in positions:
                    positions.append(length - tile_size)
            return positions

        # boundary-covering tiles which improve continuity across tile edges.
        def compute_offset_positions(length):
            offset = tile_size // 2
            positions = list(range(offset, length - tile_size + 1, tile_size))
            if (length - tile_size) not in positions:
                positions.append(length - tile_size)
            return positions
        
        y_coords = compute_positions(h)
        x_coords = compute_positions(w)

        y_covers = []
        x_covers = []
        if (h > tile_size or w > tile_size):
            y_covers = compute_offset_positions(h)
            x_covers = compute_offset_positions(w)

        return (x_coords, y_coords), (x_covers, y_covers)

    def _split_input_list_from_range(self, input_points_list, input_labels_list, x_bound, y_bound):
        local_points_list = []
        local_labels_list = []
        for points, labels in zip(input_points_list, input_labels_list):
            local_points = []
            local_labels = []
            for point, label in zip(points, labels):
                x, y = point
                if (x >= x_bound[0] and x < x_bound[1] and y >= y_bound[0] and y < y_bound[1]):
                    local_points.append([int(x - x_bound[0]), int(y - y_bound[0])])
                    local_labels.append(int(label))
            
            if (local_points):
                local_points_list.append(np.array(local_points))
                local_labels_list.append(np.array(local_labels))
        
        return local_points_list, local_labels_list
    
    def get_list_of_sam_parameters(self):
        image_list = []
        input_points_list_set = []
        input_labels_list_set = []
        for tile in self.split_tile_list:
            image_list.append(tile.get_image())
            local_points_list, local_labels_list = tile.get_input_list()
            input_points_list_set.append(local_points_list)
            input_labels_list_set.append(local_labels_list)
        
        return image_list, input_points_list_set, input_labels_list_set
    
    def combine_result_from_list(self, combine_list, save_path='./'):
       
        image = self.source_tile.get_image()
        mask_combine = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for mask, tile in zip(combine_list, self.split_tile_list):
            bound = tile.get_start_coord()
            cropped_mask = mask[:self.tile_size, :self.tile_size]
            mask_combine[bound[1]: bound[1] + self.tile_size, bound[0]: bound[0] + self.tile_size] |= mask
        
        return mask_combine

