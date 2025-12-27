@echo off
setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION
echo === Raspberry Pi Config Generator ===
echo.

set /p I_VALUE=Enter I value (stavite 00001, 00002, 00003...): 
if "%I_VALUE%"=="" (
    echo Error: I ne moze biti prazan
    goto error
)

set /p P_VALUE=Enter P value (stavite 0 za pocetak od nule): 
if "%P_VALUE%"=="" set P_VALUE=0

echo %P_VALUE% | findstr /R "^[0-9][0-9]*$" >nul
if errorlevel 1 (
    echo Error: P mora biti validan celi broj
    goto error
)

(
    echo I:%I_VALUE%
    echo P:%P_VALUE%
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