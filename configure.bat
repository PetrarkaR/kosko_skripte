@echo off
setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION
echo === Raspberry Pi Config Generator ===
echo.
set /p I_VALUE=Enter I value (stavite 00001, 00002, 00003...): 
if "%I_VALUE%"=="" (
    echo Error: I ne moze biti prazan
    goto error
)

(
    echo I:%I_VALUE%
    echo P:0
) > cnf.txt

echo.
echo cnf.txt napravljen uspesno:
type cnf.txt
echo.
pause
exit /b 0

:error
echo.
echo File not created.
pause
exit /b 1
