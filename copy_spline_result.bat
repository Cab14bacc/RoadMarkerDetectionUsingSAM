@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

REM Set the source output folder and destination
set "OUTPUT_SUBFOLDER=新生南_cropped_0_01"
set "DESTINATION_FOLDER=C:\Users\Leo\Documents\GitRepos\OSMAPI\SplineTestResults\新生南路路口"

REM Find and process all image subfolders
set J=0
for /d %%I in (".\output\%OUTPUT_SUBFOLDER%\*") do (
    set /a J+=1
    set "IMG_SUBFOLDER=%%~nxI"
    set "DEST_IMG_SUBFOLDER=!IMG_SUBFOLDER:cropped_=!"

    @REM echo Copying spline data from !IMG_SUBFOLDER!

    if not exist "%DESTINATION_FOLDER%\!DEST_IMG_SUBFOLDER!" (
        mkdir "%DESTINATION_FOLDER%\!DEST_IMG_SUBFOLDER!"
    )

    echo copy ".\output\%OUTPUT_SUBFOLDER%\!IMG_SUBFOLDER!\Spline\spline_data_result.json" ^
         "%DESTINATION_FOLDER%\!DEST_IMG_SUBFOLDER!\spline_data_result.json"

    copy ".\output\%OUTPUT_SUBFOLDER%\!IMG_SUBFOLDER!\Spline\spline_data_result.json" ^
         "%DESTINATION_FOLDER%\!DEST_IMG_SUBFOLDER!\spline_data_result.json"
)
