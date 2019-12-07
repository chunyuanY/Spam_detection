# Paper of the source codes released:
Chunyuan Yuan, Wei Zhou, Qianwen Ma, Shangwen Lv, Jizhong Han, Songlin Hu. Learning review representations from user and product level information for spam detection. In 19th IEEE International Conference on Data Mining, IEEE ICDM 2019.

# Dependencies:
Gensim==3.7.2

Jieba==0.39

Scikit-learn==0.21.2

Pytorch==1.1.0

# Datasets
The dataset can be downloaded from here: https://drive.google.com/file/d/1wFnQ_ZhpegyMoQ3lGfXiHrSHqE7ys7gl/view?usp=sharing .


# Reproduce the experimental results:
1. create an empty directory: checkpoint/
2. run script: python preprocess/preprocess.py
3. run script: python run.py


## Citation
If you find this code useful in your research, please cite our paper:
```
@inproceedings{yuan2019learning,
  title={Learning review representations from user and product level information for spam detection},
  author={Yuan, Chunyuan and Zhou, Wei and Ma, Qianwen and Lv, Shangwen and Han, Jizhong and Hu, Songlin},
  booktitle={The 19th IEEE International Conference on Data Mining},
  year={2019},
  organization={IEEE}
}
```

