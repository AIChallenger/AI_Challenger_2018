# Autonomous Driving Perception Evaluation
Autonomous Driving Perception is a task of AI Challenger 全球AI挑战赛。This python script is used for calculating the accuracy of the test result, based on your submited file and the reference file containing ground truth.  (This script is the lastest and robust version. )
# Usage
```
python autonomous_driving_perception2018_eval.py --submit SUBMIT_FILEPATH --ref REF_FILEPATH
```
A test case is provided, submited file is .zip file, reference file is ref directory, test it by:
```
python autonomous_driving_perception2018_eval.py --submit ./submission.zip --ref ./ref
```
The accuracy of the submited result, error message and warning message will be printed.
