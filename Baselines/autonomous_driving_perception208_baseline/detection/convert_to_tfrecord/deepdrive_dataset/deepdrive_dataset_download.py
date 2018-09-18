import os
import re
import zipfile
from os.path import expanduser

from .utils import mkdir_p


class DeepdriveDatasetDownload(object):

    @staticmethod
    def filter_elements(path, lambda_fn, return_relative=True, regex=None):
        """
        Filter alls elements in the path given the lambda_fn function
        :param path:
        :param lambda_fn:
        :param return_relative:
        :param regex:
        :return:
        """
        assert (regex is None or isinstance(regex, re._pattern_type))
        filtered = []
        try:
            list_dir = os.listdir(path)
        except:
            return []
        for element in list_dir:
            if not lambda_fn(os.path.join(path, element)):
                continue
            m = '' if regex is None else regex.search(element)
            if m is not None:
                filtered.append(element if return_relative else os.path.join(path, element))
        return filtered

    @staticmethod
    def filter_folders(path, return_relative=True, regex=None):
        """
        Filters all elements in the path
        :param path:
        :param return_relative:
        :param regex:
        :return:
        """
        return DeepdriveDatasetDownload.filter_elements(
            path, lambda x: os.path.isdir(x), return_relative, regex)

    @staticmethod
    def filter_files(path, return_relative=True, regex=None):
        return DeepdriveDatasetDownload.filter_elements(
            path, lambda x: os.path.isfile(x), return_relative, regex)

    @staticmethod
    def download_image_data(fold_type=None, version=None, force_download=False):
        raise NotImplementedError('')

    @staticmethod
    def download_annotation_data(fold_type=None, version=None, force_download=False):
        raise NotImplementedError('')
