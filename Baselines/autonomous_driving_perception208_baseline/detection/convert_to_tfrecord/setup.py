from setuptools import setup

from pip.req import parse_requirements

install_reqs = parse_requirements('requirements.txt')

reqs = [str(ir.req) for ir in install_reqs]

setup(
    name='deepdrive_dataset_tfrecord',
    version='0.1',
    packages=['deepdrive_dataset'],
    url='',
    license='MIT',
    author='meyerjo',
    author_email='meyerjo@tf.uni-freiburg.de',
    description='Convert Deepdrive Dataset to TFRecord Files',
    install_requires=reqs
)
