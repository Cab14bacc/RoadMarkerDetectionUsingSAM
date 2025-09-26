@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion


set "INPUT_DIR=./input/新生南_cropped_0_01"
@REM set "INPUT_DIR=./input/新生南"
set "INPUT_FILENAMES=cropped_0001.tif cropped_0002.tif cropped_0003.tif cropped_0004.tif cropped_0005.tif cropped_0006.tif cropped_0007.tif cropped_0008.tif"
@REM set "INPUT_FILENAMES=cropped_0003.tif cropped_0004.tif cropped_0005.tif cropped_0006.tif cropped_0007.tif cropped_0008.tif"
@REM set "INPUT_FILENAMES=part1_uncompressed.tif part2_uncompressed.tif"
set "CONFIG_PATH=./config_classifier.yml"
set "OUTPUT_DIR=./output/新生南_cropped_0_01"
@REM set "OUTPUT_DIR=./output/新生南"


set i=0
for %%A in (%INPUT_FILENAMES%) do (
    set "INPUT_FILENAMES[!i!]=%%A" 
    set /a i+=1
)

set /a i-=1

for /l %%J in (0, 1, !i!) do (
    @REM markerClassifier.py --image IMAGE --output OUTPUT --config CONFIG 
    set "CUR_FILE=!INPUT_FILENAMES[%%J]!"
    set "OUTPUT_SUBFOLDER=!CUR_FILE:.=_!"
    echo python markerClassifier.py --image %INPUT_DIR%/!CUR_FILE! --output %OUTPUT_DIR%/!OUTPUT_SUBFOLDER!/MarkerClassifier --config %CONFIG_PATH%
    python marker_classifier.py --image %INPUT_DIR%/!CUR_FILE! --output %OUTPUT_DIR%/!OUTPUT_SUBFOLDER!/MarkerClassifier --config %CONFIG_PATH%
)