# It's empty. Surprise!
# Please complete this by yourself.
'''
@Description: 
@Version: 1.0
@Author: Taoye Yin
@Date: 2019-08-17 15:59:28
@LastEditors: Taoye Yin
@LastEditTime: 2019-08-28 19:35:46
'''
import pandas as pd
import os
from PIL import Image
import sys
os.chdir(sys.path[0])
ROOTS = '../Dataset/'
PHASE = ['train', 'val']
SPECIES = ['rabbits', 'rats', 'chickens']  # [0,1,2]
CLASSES = ['Mammals', 'Birds'] #[0,1]
DATA_info = {'train': {'path': [], 'classes':[], 'species': []},
             'val': {'path': [], 'classes':[], 'species': []}
             }
for p in PHASE:
    for s in SPECIES:
        DATA_DIR = ROOTS + p + '\\' + s
        DATA_NAME = os.listdir(DATA_DIR) #返回指定的文件夹包含的文件或文件夹的名字的列表

        for item in DATA_NAME:
            try:
                img = Image.open(os.path.join(DATA_DIR, item))#路径拼接
            except OSError:
                pass
            else:
                DATA_info[p]['path'].append(os.path.join(DATA_DIR, item))
                if s == 'rabbits':
                    DATA_info[p]['classes'].append(0)
                    DATA_info[p]['species'].append(0)
                elif s == 'rats':
                    DATA_info[p]['classes'].append(0)
                    DATA_info[p]['species'].append(1)
                else:
                    DATA_info[p]['classes'].append(1)
                    DATA_info[p]['species'].append(2)

    ANNOTATION = pd.DataFrame(DATA_info[p])
    ANNOTATION.to_csv('Species_%s_annotation_new.csv' % p)
    print('Species_%s_annotation file_new is saved.' % p)
