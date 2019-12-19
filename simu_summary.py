"""
Summarize simulation results: MVCM-SVA pipeline
Usage: python ./simu_summary.py ./result/simu/setting_1/

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-12-28
"""

import sys
from scipy.io import loadmat
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

"""
installed all the libraries above
"""

if __name__ == '__main__':
    input_dir = sys.argv[1]

    """+++++++++++++++++++++++++++++++++++"""
    print(""" Step 0. load result """)
    beta0_file_name = input_dir + "beta0.mat"
    mat = loadmat(beta0_file_name)
    beta0 = mat['beta0']
    coord_file_name = input_dir + "coord_data.mat"
    mat = loadmat(coord_file_name)
    coord_data = mat['coord_data']
    if len(coord_data.shape) == 1:
        coord_data = coord_data.reshape(coord_data.shape[0], 1)
    elif coord_data.shape[0] == 1:
        coord_data = coord_data.T

    """+++++++++++++++++++++++++++++++++++"""
    print(""" assess the estimation accuracy """)
    nn = 200
    err_s = np.zeros(shape=(nn, 1))
    err_f = np.zeros(shape=(nn, 1))
    for ii in range(nn):
        folder_name = input_dir + str(ii+1)
        bs_file_name = folder_name + "/b_s.mat"
        mat = loadmat(bs_file_name)
        b_s = mat['b_s']
        err_s[ii] = np.sqrt(np.sum(np.mean((b_s - beta0)**2, axis=1)))
        bf_file_name = folder_name + "/b_f.mat"
        mat = loadmat(bf_file_name)
        b_f = mat['b_f']
        err_f[ii] = np.sqrt(np.sum(np.mean((b_f - beta0)**2, axis=1)))

    data_to_plot = np.hstack((err_s, err_f))
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    # Create an axes instance
    ax = fig.add_subplot(111)
    #  add patch_artist=True option to ax.boxplot() to get fill color
    bp = ax.boxplot(data_to_plot, patch_artist=True)
    #  change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set(color='#7570b3', linewidth=2)
        # change fill color
        box.set(facecolor='#1b9e77')
    #  change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    #  change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    #  change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    #  change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)
    #  Custom x-axis labels
    ax.xaxis.set_tick_params(labelsize=20)
    ax.set_xticklabels(['MVCM', 'FLFRM'])
    ax.set_ylabel('Integrated Square Error (ISE)')
    #  Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    fig_name = input_dir + "boxplot_1.pdf"
    plt.savefig(fig_name, bbox_inches='tight')
