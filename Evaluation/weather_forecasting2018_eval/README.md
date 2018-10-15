# Weather Forecasting Evaluation
Weather Forecasting is a task of AI Challenger 全球AI挑战赛。This python script is used for calculating the accuracy of the test result, based on your submited file and the reference file containing ground truth.  (This script is the lastest and robust version. )
# Usage
```
python weather_forecasting2018_eval.py --submit SUBMIT_FILEPATH --obs OBSERVATION_FILEPATH --fore RMAPS_FILEPATH
```
A test case is provided, submited file is anene.csv, observation file is obs.csv, RMAPS result is anen.csv, test it by:
```
python weather_forecasting2018_eval.py --submit ./anen.csv --obs ./obs.csv --fore ./fore.csv
```
The accuracy of the submited result, error message and warning message will be printed.

