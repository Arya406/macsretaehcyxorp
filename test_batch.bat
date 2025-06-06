@echo off
echo Testing batch script execution > test_batch_output.txt
echo Current directory: %CD% >> test_batch_output.txt
echo Python version: >> test_batch_output.txt
python --version >> test_batch_output.txt 2>&1
echo. >> test_batch_output.txt
echo Environment variables: >> test_batch_output.txt
set >> test_batch_output.txt
