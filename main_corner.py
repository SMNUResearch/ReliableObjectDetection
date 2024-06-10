import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from matplotlib.pyplot import MultipleLocator

def truncated_gaussian(mu, sigma, lower_bound, upper_bound, num_trials):
    a = (lower_bound - mu) / sigma
    b = (upper_bound - mu) / sigma

    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=num_trials)

def get_IoU(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(x2 - x1, 0) * max(y2 - y1, 0)
    union = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) + (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) - intersection

    IoU = intersection / union

    return IoU

def corner_simulation(mu, sigma, r_min, r_max, num_trials, bbox_pos, uncertain_pos=[False, False, False, False]):
    d1 = truncated_gaussian(mu, sigma, r_min, r_max, num_trials)
    d2 = truncated_gaussian(mu, sigma, r_min, r_max, num_trials)
    d3 = truncated_gaussian(mu, sigma, r_min, r_max, num_trials)
    d4 = truncated_gaussian(mu, sigma, r_min, r_max, num_trials)

    if uncertain_pos[0]:
        xmin_diff = d1
    else:
        xmin_diff = np.zeros(num_trials)

    if uncertain_pos[1]:
        ymin_diff = d2
    else:
        ymin_diff = np.zeros(num_trials)

    if uncertain_pos[2]:
        xmax_diff = d3
    else:
        xmax_diff = np.zeros(num_trials)

    if uncertain_pos[3]:
        ymax_diff = d4
    else:
        ymax_diff = np.zeros(num_trials)

    # scale
    w_diff = (xmax_diff - xmin_diff) * 0.5
    h_diff = (ymax_diff - ymin_diff) * 0.5
    # shift
    center_x_diff = (xmax_diff + xmin_diff) * 0.5
    center_y_diff = (ymax_diff + ymin_diff) * 0.5

    # bbox_pos: [center_x, center_y, w, h];
    new_center_x = (bbox_pos[0] + center_x_diff)
    new_center_y = (bbox_pos[1] + center_y_diff)
    new_w = (bbox_pos[2] + w_diff)
    new_h = (bbox_pos[3] + h_diff)

    # IoU
    IoU_list = []
    bbox_pos = np.array([bbox_pos[0] - bbox_pos[2], bbox_pos[1] - bbox_pos[3], bbox_pos[0] + bbox_pos[2], bbox_pos[1] + bbox_pos[3]])
    for i in range(0, num_trials):
        new_bbox_pos = np.array([new_center_x[i] - new_w[i], new_center_y[i] - new_h[i], new_center_x[i] + new_w[i], new_center_y[i] + new_h[i]])
        IoU = get_IoU(bbox_pos, new_bbox_pos)
        IoU_list.append(IoU)

    return w_diff, h_diff, center_x_diff, center_y_diff, np.array(IoU_list)

if __name__ == '__main__':
    # set seed
    seed = 1000
    np.random.seed(seed)

    # set options
    mu = 0
    sigma = 1
    r_min = -3
    r_max = 3
    num_trials = 100000
    num_bins = 100

    # set bbox
    bbox_pos_list = [[320, 320, 20, 20], [320, 320, 50, 50], [320, 320, 100, 100]]
    uncertain_pos_list = []
    for xmin in [True, False]:
        for ymin in [True, False]:
            for xmax in [True, False]:
                for ymax in [True, False]:
                    uncertain_pos_list.append([xmin, ymin, xmax, ymax])

    # run simulation & save visualization results
    parser = argparse.ArgumentParser()
    parser.add_argument('-OPTION', type=str, default='scale') # ['scale', 'shift', 'IoU']
    opt = parser.parse_args()

    simulation_option = opt.OPTION

    if simulation_option == 'scale':
        output_path = './corner_scale/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        for i in range(0, len(uncertain_pos_list)):
            results = corner_simulation(mu, sigma, r_min, r_max, num_trials, bbox_pos_list[1], uncertain_pos=uncertain_pos_list[i])
            plt.figure(figsize=(3, 3))
            plt.hist2d(results[0], results[1], bins=(num_bins, num_bins), cmap='magma')
            ax = plt.gca()

            x_uncertain = uncertain_pos_list[i][0] + uncertain_pos_list[i][2]
            y_uncertain = uncertain_pos_list[i][1] + uncertain_pos_list[i][3]

            if x_uncertain == 2:
                x_step = MultipleLocator(1)
            else:
                x_step = MultipleLocator(0.5)

            if y_uncertain == 2:
                y_step = MultipleLocator(1)
            else:
                y_step = MultipleLocator(0.5)

            ax.xaxis.set_major_locator(x_step)
            ax.yaxis.set_major_locator(y_step)

            info = uncertain_pos_list[i]
            output_file = output_path + 'scale_' + str(int(info[0])) + '_' + str(int(info[1])) + '_' + str(int(info[2])) + '_' + str(int(info[3])) + '.pdf'
            plt.savefig(output_file, bbox_inches='tight')
    elif simulation_option == 'shift':
        output_path = './corner_shift/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        for i in range(0, len(uncertain_pos_list)):
            results = corner_simulation(mu, sigma, r_min, r_max, num_trials, bbox_pos_list[1], uncertain_pos=uncertain_pos_list[i])
            plt.figure(figsize=(3, 3))
            plt.hist2d(results[2], results[3], bins=(num_bins, num_bins), cmap='magma')
            ax = plt.gca()

            x_uncertain = uncertain_pos_list[i][0] + uncertain_pos_list[i][2]
            y_uncertain = uncertain_pos_list[i][1] + uncertain_pos_list[i][3]

            if x_uncertain == 2:
                x_step = MultipleLocator(1)
            else:
                x_step = MultipleLocator(0.5)

            if y_uncertain == 2:
                y_step = MultipleLocator(1)
            else:
                y_step = MultipleLocator(0.5)

            ax.xaxis.set_major_locator(x_step)
            ax.yaxis.set_major_locator(y_step)

            info = uncertain_pos_list[i]
            output_file = output_path + 'shift_' + str(int(info[0])) + '_' + str(int(info[1])) + '_' + str(int(info[2])) + '_' + str(int(info[3])) + '.pdf'
            plt.savefig(output_file, bbox_inches='tight')
    elif simulation_option == 'IoU':
        output_path = './corner_iou/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # COCO standard (bbox area) -> 'small': [0 ** 2, 32 ** 2]; 'medium': [32 ** 2, 96 ** 2]; 'large': [96 ** 2, 1e5 ** 2];
        bins_count_list = []
        cdf_list = []
        for i in range(0, len(bbox_pos_list)):
            for j in range(0, len(uncertain_pos_list)):
                results = corner_simulation(mu, sigma, r_min, r_max, num_trials, bbox_pos_list[i], uncertain_pos=uncertain_pos_list[j])
                count, bins_count = np.histogram(results[4], bins=num_bins)
                pdf = count / sum(count)
                cdf = np.cumsum(pdf)
                bins_count_list.append(bins_count)
                cdf_list.append(cdf)

        # small
        plt.figure(figsize=(3, 3))
        plt.plot(bins_count_list[0][1:], cdf_list[0], label='4')
        plt.plot(bins_count_list[1][1:], cdf_list[1], label='3')
        plt.plot(bins_count_list[3][1:], cdf_list[3], label='2')
        plt.plot(bins_count_list[7][1:], cdf_list[7], label='1')
        plt.legend()
        plt.grid()
        output_file = output_path + 'IoU_samll.pdf'
        plt.savefig(output_file, bbox_inches='tight')

        # medium
        plt.figure(figsize=(3, 3))
        plt.plot(bins_count_list[16][1:], cdf_list[16], label='4')
        plt.plot(bins_count_list[17][1:], cdf_list[17], label='3')
        plt.plot(bins_count_list[19][1:], cdf_list[19], label='2')
        plt.plot(bins_count_list[23][1:], cdf_list[23], label='1')
        plt.legend()
        plt.grid()
        output_file = output_path + 'IoU_medium.pdf'
        plt.savefig(output_file, bbox_inches='tight')

        # large
        plt.figure(figsize=(3, 3))
        plt.plot(bins_count_list[32][1:], cdf_list[32], label='4')
        plt.plot(bins_count_list[33][1:], cdf_list[33], label='3')
        plt.plot(bins_count_list[35][1:], cdf_list[35], label='2')
        plt.plot(bins_count_list[39][1:], cdf_list[39], label='1')
        plt.legend()
        plt.grid()
        output_file = output_path + 'IoU_large.pdf'
        plt.savefig(output_file, bbox_inches='tight')

        # object scale
        plt.figure(figsize=(3, 3))
        plt.plot(bins_count_list[0][1:], cdf_list[0], label='small')
        plt.plot(bins_count_list[16][1:], cdf_list[16], label='medium')
        plt.plot(bins_count_list[32][1:], cdf_list[32], label='large')
        plt.legend()
        plt.grid()
        output_file = output_path + 'IoU_scale.pdf'
        plt.savefig(output_file, bbox_inches='tight')

