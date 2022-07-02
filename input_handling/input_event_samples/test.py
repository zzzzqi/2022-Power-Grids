import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from enum import Enum
import shutil

# signal = pd.read_csv('input_event_sample00.csv', index_col=4)
# plt.xlim(-2, 2, 1)
# plt.ylim(-2, 2, 1)
# plt.axis('on')
# plt.plot(signal['Vab'], np.roll(signal['Vab'], 20))
# plt.show()

# class Waveform(Enum):
#     Vab = 'Vab'
#     Vbc = 'Vbc'
#     Vca = 'Vca'
#     Ia = 'Ia'
#     Ib = 'Ib'
#     Ic = 'Ic'

# import_file = 'input_event_sample00.csv'
# export_image = import_file[:len(import_file)-4]
# print(export_image)
# current_directory = os.getcwd()
# files = os.listdir(current_directory)
# for file in files:
#     print(file)

current_directory = os.getcwd()
files = os.listdir(current_directory)
for file in files:
    if re.search('.png$', file) is not None:
        os.remove(file)	
    if os.path.isdir(file):
        shutil.rmtree(file)
