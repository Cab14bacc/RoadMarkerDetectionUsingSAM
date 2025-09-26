@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion


set "INPUT_MASK_PARENT_DIR=./output/新生南_cropped_0_01"
@REM set "INPUT_MASK_PARENT_DIR=./output/新生南"
set "MASK_SUBFOLDERS=cropped_0001_tif cropped_0002_tif cropped_0003_tif cropped_0004_tif cropped_0005_tif cropped_0006_tif cropped_0007_tif cropped_0008_tif"
@REM set "MASK_SUBFOLDERS=cropped_0007_tif cropped_0008_tif"

set "INPUT_IMAGE_DIR=./input/新生南_cropped_0_01"
set "INPUT_IMAGE_FILENAMES=cropped_0001.tif cropped_0002.tif cropped_0003.tif cropped_0004.tif cropped_0005.tif cropped_0006.tif cropped_0007.tif cropped_0008.tif"
@REM set "INPUT_IMAGE_FILENAMES=cropped_0007_tif cropped_0008_tif"


@REM set "MASK_SUBFOLDERS=part1_uncompressed_tif part2_uncompressed_tif"
set "CONFIG_PATH=./config_0_01.yml"
set "OUTPUT_PARENT_DIR=./output/新生南_cropped_0_01"
@REM set "OUTPUT_PARENT_DIR=./output/新生南"


set i=0
for %%A in (%MASK_SUBFOLDERS%) do (
    set "MASK_SUBFOLDERS[!i!]=%%A" 
    set /a i+=1
)

set i=0
for %%A in (%INPUT_IMAGE_FILENAMES%) do (
    set "INPUT_IMAGE_FILENAMES[!i!]=%%A" 
    set /a i+=1
)

set /a i-=1

for /l %%J in (0, 1, !i!) do (
    @REM python spline_test.py -i <image_file_path> -o <output_path>
    set "INPUT_MASK_DIR=!INPUT_MASK_PARENT_DIR!/!MASK_SUBFOLDERS[%%J]!"
    set "INPUT_MASK_PATH=!INPUT_MASK_DIR!/Predictor/line/tile_combine.png"
    set "INPUT_IMAGE_PATH=!INPUT_IMAGE_DIR!/!INPUT_IMAGE_FILENAMES[%%J]!"
    @REM set "INPUT_MASK_PATH=!INPUT_MASK_DIR!/Predictor/line/map_data.json"
    set "OUTPUT_DIR=!OUTPUT_PARENT_DIR!/!MASK_SUBFOLDERS[%%J]!/Spline"
    
    echo python spline_test.py -i !INPUT_IMAGE_PATH! -m !INPUT_MASK_PATH! -o !OUTPUT_DIR! -c %CONFIG_PATH%
    python spline_test.py -i !INPUT_IMAGE_PATH! -m !INPUT_MASK_PATH! -o !OUTPUT_DIR! -c %CONFIG_PATH%
    
)