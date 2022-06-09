import os
import re
from PIL import Image
from converter_python import plot_phase_space_graph

# check if the phaseSpace folder exist
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('create successful')

# file path
data_path = os.getcwd()  # get current folder
# phase_space_path_colour = data_path + os.sep + "phaseSpace_colour"
phase_space_path_gray = data_path + os.sep + 'phaseSpace_gray'

# check if the dir exists, if not, create a new folder
# mkdir(phase_space_path_colour)
mkdir(phase_space_path_gray)

# reg pattern
filename_prefix = 'flickers'                                                    ## TO MODIFY
filename_suffix = '.csv$'
# png_colour_suffix = '_colour.png'
png_gray_suffix = '_gray.png'

# Batch processing
file_list = os.listdir(data_path)
for file in file_list:
    # check if the file is a .csv file
    if re.search(filename_prefix, file) is not None and re.search(filename_suffix, file) is not None:
        file_path = data_path + os.sep + file
        # export_colour_path = phase_space_path_colour + os.sep + re.sub(csv_suffix, png_colour_suffix, file)
        export_gray_path = phase_space_path_gray + os.sep + re.sub(filename_suffix, png_gray_suffix, file)
        plot_phase_space_graph(file_path, export_gray_path, 20)                 ## TAU input is 20
        # colour_image = Image.open(export_colour_path)
        # gray_image = colour_image.convert('L')
        # gray_image.save(export_gray_path)
