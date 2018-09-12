#!/user/bin/env python
# -*- coding:utf-8 -*-

from data_process import seg_words, load_data_from_csv
import config
import logging
import argparse
from sklearn.externals import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', '--model_name', type=str, nargs='?',
                        help='the name of model')

    args = parser.parse_args()
    model_name = args.model_name
    if not model_name:
        model_name = "model_dict.pkl"

    # load data
    logger.info("start load data")
    test_data_df = load_data_from_csv(config.test_data_path)

    # load model
    logger.info("start load model")
    classifier_dict = joblib.load(config.model_save_path + model_name)

    columns = test_data_df.columns.tolist()
    # seg words
    logger.info("start seg test data")
    content_test = test_data_df.iloc[:, 1]
    content_test = seg_words(content_test)
    logger.info("complete seg test data")

    # model predict
    logger.info("start predict test data")
    for column in columns[2:]:
        test_data_df[column] = classifier_dict[column].predict(content_test)
        logger.info("compete %s predict" % column)

    test_data_df.to_csv(config.test_data_predict_out_path, encoding="utf_8_sig", index=False)
    logger.info("compete predict test data")
