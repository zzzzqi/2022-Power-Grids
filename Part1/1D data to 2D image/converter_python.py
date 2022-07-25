import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# generate table which consist of x, y which y delay tau of x
def generate_2d_phase_space(table, tau):
    ps = table[['amplitude (in pu)']].copy()
    ps.rename(columns={'amplitude (in pu)': 'x'}, inplace=True)
    ps['y'] = np.roll(table['amplitude (in pu)'], tau)
    return ps


# only plot phase space image
def plot_phase_space_graph(import_file, export_file, tau):
    # load data
    signal = pd.read_csv(import_file, index_col=0)  # index_col means choose which col as the row labels
    pf = generate_2d_phase_space(signal, 30)
    # plot
    fig = plt.figure()
    fig.set_size_inches(25.6, 25.6)
    plt.style.use('grayscale')  # plot the graph with grayscale
    plt.plot(pf['x'], pf['y'])
    # plt.xlim(-2, 2)
    # plt.ylim(-2, 2)
    plt.axis('off')
    # plt.show()
    plt.savefig(export_file, dpi=10)


# only plot signal image
def plot_time_series_graph(import_file, export_file, tau):
    # load data
    signal = pd.read_csv("import_file", index_col=0)  # index_col means choose which col as the row labels
    # plot
    plt.figure()
    plt.plot(signal['timestamp (in ms)'], signal['amplitude (in pu)'])
    plt.axis('on')
    # plt.show()
    plt.savefig('export_file')


# plot signal image & phase space image
# pf = generate_2D_phase_space(signal, 30)
#
# fig, axs = plt.subplots(2)
# fig.suptitle('Vertically stacked subplots')
# axs[0].plot(signal['timestamp (in ms)'], signal['amplitude (in pu)'])
# axs[1].plot(pf['x'], pf['y'])
# axs[1].set_xlim(-2, 2)
# axs[1].set_ylim(-2, 2)
# plt.show()

# data information
# print(signal.dtypes) print the data type of the table
# print(signal.info) print the first and last five rows of the table and the number of the rows and cols

