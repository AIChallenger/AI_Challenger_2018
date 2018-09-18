
import json
import argparse
import time


def __load_data(submit_file, reference_file):
  # load submit result and reference result

    with open(submit_file, 'r') as file1:
        submit_data = json.load(file1)
    with open(reference_file, 'r') as file1:
        ref_data = json.load(file1)
    if len(submit_data) != len(ref_data):
        result['warning'].append('Inconsistent number of images between submission and reference data \n')
    submit_dict = {}
    ref_dict = {}
    for item in submit_data:
        submit_dict[item['image_id']] = item['label_id']
    for item in ref_data:
        ref_dict[item['image_id']] = int(item['label_id'])
    return submit_dict, ref_dict


def __eval_result(submit_dict, ref_dict):
    # eval accuracy

    right_count = 0
    for (key, value) in ref_dict.items():

        if key not in set(submit_dict.keys()):
            result['warning'].append('lacking image %s in your submission file \n' % key)
            print('warnning: lacking image %s in your submission file' % key)
            continue

        if value in submit_dict[key][:3]:
            right_count += 1

    result['score'] = str(float(right_count)/max(len(ref_dict), 1e-5))
    return result


if __name__ == '__main__':

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument(
        '--submit',
        type=str,
        default='./submit.json',
        help="""\
        Path to submission file\
        """
    )

    PARSER.add_argument(
        '--ref',
        type=str,
        default='./ref.json',
        help="""\
        Path to reference file\
        """
    )

    FLAGS = PARSER.parse_args()

    result = {'error': [], 'warning': [], 'score': None}

    START_TIME = time.time()
    SUBMIT = {}
    REF = {}

    try:
        SUBMIT, REF = __load_data(FLAGS.submit, FLAGS.ref)
    except Exception as error:
        result['error'].append(str(error))
    try:
        result = __eval_result(SUBMIT, REF)
    except Exception as error:
        result['error'].append(str(error))
    print('Evaluation time of your result: %f s' % (time.time() - START_TIME))

print(result)
