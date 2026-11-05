@echo off
cd /d "C:\Users\HP\OneDrive\Documents\fake news detection"

start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:5000"

venv\Scripts\python.exe app.py

pause