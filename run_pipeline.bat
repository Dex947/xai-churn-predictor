@echo off
echo ================================================================================
echo Customer Churn Prediction System - Training Pipeline
echo ================================================================================
echo.

python main.py --skip-download

echo.
echo ================================================================================
echo Pipeline Complete! Check the following locations:
echo - Models: data/models/
echo - Visualizations: data/plots/
echo - Reports: data/results/
echo ================================================================================
echo.
echo To launch the Streamlit dashboard, run: run_dashboard.bat
pause
