import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import ast
import glob
import shutil
import sys
import numpy as np
import imagesize
from tqdm.notebook import tqdm

git clone https://github.com/AlexeyAB/darknet.git

%cd darknet

!cp '../../input/libcuda/libcuda.so' .

!sed -i 's/OPENCV=0/OPENCV=1/g' Makefile
!sed -i 's/GPU=0/GPU=1/g' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/g' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/g' Makefile
!sed -i 's/LIBSO=0/LIBSO=1/' Makefile
!sed -i "s/ARCH= -gencode arch=compute_60,code=sm_60/ARCH= ${ARCH_VALUE}/g" Makefile

!sed -i 's/LDFLAGS+= -L\/usr\/local\/cuda\/lib64 -lcuda -lcudart -lcublas -lcurand/LDFLAGS+= -L\/usr\/local\/cuda\/lib64 -lcudart -lcublas -lcurand -L\/kaggle\/working\/darknet -lcuda/' Makefile
!make &> compile.log