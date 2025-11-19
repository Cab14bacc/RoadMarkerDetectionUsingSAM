import numpy as np
import cv2
from .tileData import SamTileData
from .tileManagerInterface import TileManagerInterface
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
class SamTileManager(TileManagerInterface):
    def __init__(self, source_image, input_points_list, input_labels_list, tile_size=1024):

        self.source_tile = SamTileData(
            image=source_image,
            start_coord=[0, 0],
            input_points_list=input_points_list,
            input_labels_list=input_labels_list,
            tile_size=tile_size
        )

        self.tile_size = tile_size
        self.split_tile_list = []

    def split_tile(self):
        h, w, _ = self.source_tile.get_image().shape
        coords, overlap_coords = self.split_tile_position(h, w)
        
        self.split_tile_list = []

        self._collect_tiles(coords)
        self._collect_tiles(overlap_coords)

    def _collect_tiles(self, tile_coords):
        image = self.source_tile.get_image()
        input_points_list, input_labels_list = self.source_tile.get_input_list()

        tile_size = self.tile_size

        h, w, _ = image.shape
        count = 0 
        for x_start, y_start in tile_coords:
            x_end = min(x_start + tile_size, w)
            y_end = min(y_start + tile_size, h)
            tile = image[y_start:y_end, x_start:x_end].copy()
            # cv2.imshow("sldfkj", tile)
            # cv2.waitKey()

            local_points_list, local_labels_list = self._split_input_list_from_range(input_points_list, input_labels_list, [x_start, x_end], [y_start, y_end])


            if (local_points_list):
                tile_data = SamTileData(
                    image=tile,
                    start_coord=[x_start, y_start],
                    input_points_list=local_points_list,
                    input_labels_list=local_labels_list,
                    tile_size=tile_size
                )
                self.split_tile_list.append(tile_data)

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
    
    def combine_result_from_list(self, combine_list):
       
        image = self.source_tile.get_image()
        mask_combine = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for mask, tile in zip(combine_list, self.split_tile_list):
            bound = tile.get_start_coord()
            h, w = mask.shape
            mask_combine[bound[1]: bound[1] + h, bound[0]: bound[0] + w] |= mask
        
        return mask_combine

