@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion


set "INPUT_DIR=./input/新生南_cropped_0_01"
@REM set "INPUT_DIR=./input/新生南"
set "INPUT_FILENAMES=cropped_0001.tif cropped_0002.tif cropped_0003.tif cropped_0004.tif cropped_0005.tif cropped_0006.tif cropped_0007.tif cropped_0008.tif"
@REM set "INPUT_FILENAMES=cropped_0005.tif cropped_0003.tif"
@REM set "INPUT_FILENAMES=part1_uncompressed.tif part2_uncompressed.tif"
set "CONFIG_PATH=./config_0_01.yml"
set "OUTPUT_DIR=./output/新生南_cropped_0_01"
@REM set "OUTPUT_DIR=./output/新生南"

set i=0
for %%A in (%INPUT_FILENAMES%) do (
    set "INPUT_FILENAMES[!i!]=%%A" 
    set /a i+=1
)

set /a i-=1

for /l %%J in (0, 1, !i!) do (
    @REM python predictor.py -i <image_file_path> -o <output_path> --config <config_path> --usage <usage> --bigtiff
    set "CUR_FILE=!INPUT_FILENAMES[%%J]!"
    set "OUTPUT_SUBFOLDER=!CUR_FILE:.=_!"
    echo python predictor.py -i %INPUT_DIR%/!CUR_FILE! -o %OUTPUT_DIR%/!OUTPUT_SUBFOLDER!/Predictor --config %CONFIG_PATH% --negative-mask %OUTPUT_DIR%/!OUTPUT_SUBFOLDER!/MarkerClassifier/result.png --usage line
    @REM echo python predictor.py -i %INPUT_DIR%/!CUR_FILE! -o %OUTPUT_DIR%/!OUTPUT_SUBFOLDER!/Predictor --config %CONFIG_PATH% --negative-mask %OUTPUT_DIR%/!OUTPUT_SUBFOLDER!/MarkerClassifier/result.png --usage line
    python predictor.py -i %INPUT_DIR%/!CUR_FILE! -o %OUTPUT_DIR%/!OUTPUT_SUBFOLDER!/Predictor --config %CONFIG_PATH%  --negative-mask %OUTPUT_DIR%/!OUTPUT_SUBFOLDER!/MarkerClassifier/result.png --usage line
    @REM python predictor.py -i %INPUT_DIR%/!CUR_FILE! -o %OUTPUT_DIR%/!OUTPUT_SUBFOLDER!/Predictor --config %CONFIG_PATH%  --negative-mask %OUTPUT_DIR%/!OUTPUT_SUBFOLDER!/Segmentator/negative_mask.png --usage line
)