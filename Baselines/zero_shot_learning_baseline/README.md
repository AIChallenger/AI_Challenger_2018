### Description
This is the baseline method (MDP) for AI_Challenger Zero-shot Learning Competition.

### Citing
If you use the data and code, please cite the following papers:

"AI Challenger : A Large-scale Dataset for Going Deeper in Image Understanding".
Find the dataset paper here.   
[Find the dataset paper here.](https://arxiv.org/abs/1711.06475)

```
@article{wu2017ai,
  title={AI Challenger: A Large-scale Dataset for Going Deeper in Image Understanding},
  author={Wu, Jiahong and Zheng, He and Zhao, Bo and Li, Yixin and Yan, Baoming and Liang, Rui and Wang, Wenjia and Zhou, Shipei and Lin, Guosen and Fu, Yanwei and others},
  journal={arXiv preprint arXiv:1711.06475},
  year={2017}
}
```

"Zero-shot Learning Posed as a Missing Data Problem".
Find the methodology paper here.   
[Find the methodology paper here.](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w38/Zhao_Zero-Shot_Learning_Posed_ICCV_2017_paper.pdf)

```
@inproceedings{zhao2017zero,
  title={Zero-shot learning posed as a missing data problem},
  author={Zhao, Bo and Wu, Botong and Wu, Tianfu and Wang, Yizhou},
  booktitle={Proceedings of the International Conference on Computer Vision Workshops},
  pages={2616--2622},
  year={2017}
}
```

### Environment Setup
1. Make sure that python 3.6, sklearn, tensorflow1.4.0 and keras are installed.
1. Download the data and code.
1. Put the data and code in the following directory:
```
-Baselines
--zsl_a_animals_train_20180321
--zsl_a_animals_test_20180321
--zero_shot_learning_baseline
```

### Running
Train the feature extractor (CNN):
```bash
python train_CNN.py Animals True 0.05
```

Extract image features:
```bash
python feature_extract.py Animals model/mobile_Animals_wgt.h5
```

Implement zero-shot learning:
```bash
python MDP.py Animals
```

