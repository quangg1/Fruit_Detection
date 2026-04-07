@echo off
cd /d "%~dp0"
echo Dat FRUIT_CAMERA truoc neu dung IP Webcam, vi du:
echo   set FRUIT_CAMERA=http://192.168.1.10:8080/video
python -m fruit_project.live
pause
