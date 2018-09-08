# Automated Segmentation of Retinal Edema Lesions Evaluation
Automated Segmentation of Retinal Edema Lesions is a task of AI Challenger 全球AI挑战赛。This python script is used for calculating the accuracy of the test result, based on your submited file and the reference file containing ground truth.  (This script is the lastest and robust version. )
# Usage
```
python fundus_lesion2018_eval.py --submit SUBMIT_FILEPATH --ref REF_FILEPATH
```
A test case is provided, submited file is pred.txt, reference file is ans.txt, test it by:
```
python fundus_lesion2018_eval.py --submit submission_example.zip --ref groundtruth_example
```
The accuracy of the submited result, error message and warning message will be printed.

# Update Record
## 2018-09-08 23:00:00
Because of real environment is much different from example, so there may be some errors, and we are tring to solve it.

