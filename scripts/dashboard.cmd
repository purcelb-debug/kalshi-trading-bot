@echo off
rem Opens the Kalshi bot dashboard at http://localhost:8501 (local only).
cd /d "%~dp0.."
".venv\Scripts\python.exe" -m streamlit run executor\dashboard.py --server.headless false
