import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def vis_metric(result_file, output_path):
    x = np.load(result_file)

    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=x[:, 2], rasterized=True)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    limit = max(np.abs(np.max(x[:, 2])), np.abs(np.min(x[:, 2])))
    ax.set_zlim3d(-limit, limit)
    ax.set_zticks([0])
    plt.savefig(output_path + (result_file.split('.')[-2]).split('/')[-1] + '.pdf', bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-OPTION', type=str, default='localization') # ['localization']
    opt = parser.parse_args()
    option = opt.OPTION

    output_path = './vis_' + option + '/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for result_file in sorted(os.listdir('./' + option + '/')):
        vis_metric('./' + option + '/' + result_file, output_path)

