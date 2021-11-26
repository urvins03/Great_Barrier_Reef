import zipfile
import shutil
import os
import tensorflow as tf
import pandas as pd
import numpy as np

path_reef = 'E:/kaggle_great_barrier_reef/Great_Barrier_Reef/'
zip_file  = 'tensorflow-great-barrier-reef.zip'
data_path = 'data_reef'

with zipfile.ZipFile(path_reef+zip_file, 'r') as zip_ref:
    zip_ref.extractall(os.path.join(path_reef,data_path))