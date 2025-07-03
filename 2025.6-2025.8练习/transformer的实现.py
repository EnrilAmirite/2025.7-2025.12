# 这里用transformer架构实现一个情感分类模型。。。
# 数据库是经典的IMDB
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Counter
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
