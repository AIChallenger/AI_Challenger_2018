# Plant Disease Recognition Evaluation
Plant Disease Recognition is a task of AI Challenger 全球AI挑战赛。This python script is used for calculating the accuracy of the test result, based on your submited file and the reference file containing ground truth.  (This script is the lastest and robust version. )
# Usage
```
python plant_disease_recognition2018_eval.py --submit SUBMIT_FILEPATH --ref REF_FILEPATH
```
A test case is provided, submited file is prediction_example.json, reference file is ref_example.json, test it by:
```
python plant_disease_recognition2018_eval.py --submit prediction_example.json --ref ref_example.json
```
The accuracy of the submited result, error message and warning message will be printed.