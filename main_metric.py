import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def generate_detection(num_detections, num_tp, min_confidence, IoU_threshold):
    # 0: TP or FP
    # 1: confidence score
    # 2: IoU
    detection = np.zeros((num_detections, 3))
    x = np.zeros(num_detections)
    x[0:num_tp] = 1.0
    np.random.shuffle(x)
    detection[:, 0] = x

    for i in range(0, num_detections):
        detection[i][1] = 1.0 - (1.0 - min_confidence) / num_detections * (i + 1)
        if detection[i][0] == 1:
            detection[i][2] = np.random.uniform(low=IoU_threshold, high=1.0)
        else:
            detection[i][2] = np.random.uniform(low=0.0, high=IoU_threshold)

    return detection

def get_precision_recall(detection, num_gt):
    tp = np.cumsum(detection[:, 0])
    fp = np.cumsum(1 - detection[:, 0])
    recall = tp / num_gt
    precision = tp / (tp + fp)

    return precision, recall

def get_AP_11(precision, recall):
    AP = 0

    recall_k = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.1)) + 1, endpoint=True)
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    recall_index = np.searchsorted(recall, recall_k, side='left')
    try:
        for r_i, p_i in enumerate(recall_index):
            AP += precision[p_i]
    except:
        pass

    AP = AP / 11

    return AP

def get_AP_101(precision, recall):
    AP = 0
    recall_k = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    recall_index = np.searchsorted(recall, recall_k, side='left')
    try:
        for r_i, p_i in enumerate(recall_index):
            AP += precision[p_i]
    except:
        pass

    AP = AP / 101

    return AP

def get_AP_all(precision, recall):
    AP = 0
    r = np.concatenate(([0.], recall, [1.]))
    p = np.concatenate(([0.], precision, [0.]))

    for i in range(len(p) - 1, 0, -1):
        p[i - 1] = np.maximum(p[i - 1], p[i])

    index = np.where(r[1:] != r[:-1])[0]
    AP = np.sum((r[index + 1] - r[index]) * p[index + 1])

    return AP

def get_LRP(detection, num_gt, IoU_threshold):
    num_tp = np.sum(detection[:, 0])
    num_detection = detection.shape[0]
    num_fp = num_detection - num_tp
    num_fn = num_gt - num_tp

    sum_IoU = np.sum(detection[np.where(detection[:, 0] == 1), 2])

    lrp = ((num_tp - sum_IoU) / (1 - IoU_threshold) + num_fp + num_fn) / (num_tp + num_fp + num_fn)

    return lrp

def simulate_localization(num_trials, num_detections, num_tp, num_gt, min_confidence, IoU_threshold, output_path, mode=0):
    ap_11_change = []
    ap_101_change = []
    ap_all_change = []
    lrp_change = []

    for i in range(0, num_trials):
        detection = generate_detection(num_detections, num_tp, min_confidence, IoU_threshold)
        precision, recall = get_precision_recall(detection, num_gt)
        ap_11 = get_AP_11(precision, recall)
        ap_101 = get_AP_101(precision, recall)
        ap_all = get_AP_all(precision, recall)
        lrp = get_LRP(detection, num_gt, IoU_threshold)

        tp_index = np.where(detection[:, 0] == 1)[0]
        fp_index = np.where(detection[:, 0] == 0)[0]

        if mode == 2 or mode == 3: # TP -> FP;
            alpha = np.random.choice(tp_index)
            new_detection = np.copy(detection)
            new_detection[alpha, 0] = 0
            new_detection[alpha, 2] = np.random.uniform(low=0.0, high=IoU_threshold)
            if mode == 3:
                if len(fp_index[np.where(fp_index > alpha)]) > 0:
                    beta = np.random.choice(fp_index[np.where(fp_index > alpha)])
                    new_detection[beta, 0] = 1
                    new_detection[beta, 2] = np.random.uniform(low=IoU_threshold, high=1.0)
        elif mode == 5 or mode == 6: # FP -> TP;
            alpha = np.random.choice(fp_index)
            new_detection = np.copy(detection)
            new_detection[alpha, 0] = 1
            new_detection[alpha, 2] = np.random.uniform(low=IoU_threshold, high=1.0)
            if mode == 6:
                if len(tp_index[np.where(tp_index > alpha)]) > 0:
                    beta = np.random.choice(tp_index[np.where(tp_index > alpha)])
                    new_detection[beta, 0] = 0
                    new_detection[beta, 2] = np.random.uniform(low=0.0, high=IoU_threshold)

        precision, recall = get_precision_recall(new_detection, num_gt)
        new_ap_11 = get_AP_11(precision, recall)
        new_ap_101 = get_AP_101(precision, recall)
        new_ap_all = get_AP_all(precision, recall)
        new_lrp = get_LRP(new_detection, num_gt, IoU_threshold)

        info_ap_11 = np.zeros(3)
        info_ap_11[0] = detection[alpha][1]
        info_ap_11[1] = detection[alpha][2]
        info_ap_11[2] = new_ap_11 - ap_11

        info_ap_101 = np.zeros(3)
        info_ap_101[0] = detection[alpha][1]
        info_ap_101[1] = detection[alpha][2]
        info_ap_101[2] = new_ap_101 - ap_101

        info_ap_all = np.zeros(3)
        info_ap_all[0] = detection[alpha][1]
        info_ap_all[1] = detection[alpha][2]
        info_ap_all[2] = new_ap_all - ap_all

        info_lrp = np.zeros(3)
        info_lrp[0] = detection[alpha][1]
        info_lrp[1] = detection[alpha][2]
        info_lrp[2] = new_lrp - lrp

        ap_11_change.append(info_ap_11)
        ap_101_change.append(info_ap_101)
        ap_all_change.append(info_ap_all)
        lrp_change.append(info_lrp)

    ap_11_change = np.array(ap_11_change)
    ap_101_change = np.array(ap_101_change)
    ap_all_change = np.array(ap_all_change)
    lrp_change = np.array(lrp_change)

    # save results
    np.save(output_path + str(mode) + '_ap_11_change.npy', ap_11_change)
    np.save(output_path + str(mode) + '_ap_101_change.npy', ap_101_change)
    np.save(output_path + str(mode) + '_ap_all_change.npy', ap_all_change)
    np.save(output_path + str(mode) + '_lrp_change.npy', lrp_change)

def simulate_existence(num_trials, num_detections, num_tp, num_gt, min_confidence, IoU_threshold, output_path, mode=0):
    ap_11_change = []
    ap_101_change = []
    ap_all_change = []
    lrp_change = []

    for i in range(0, num_trials):
        detection = generate_detection(num_detections, num_tp, min_confidence, IoU_threshold)
        precision, recall = get_precision_recall(detection, num_gt)
        ap_11 = get_AP_11(precision, recall)
        ap_101 = get_AP_101(precision, recall)
        ap_all = get_AP_all(precision, recall)
        lrp = get_LRP(detection, num_gt, IoU_threshold)

        tp_index = np.where(detection[:, 0] == 1)[0]
        fp_index = np.where(detection[:, 0] == 0)[0]

        if mode == 2: # FP -> TP;
            update_num_gt = num_gt + 1
            alpha = np.random.choice(fp_index)
            new_detection = np.copy(detection)
            new_detection[alpha, 0] = 1
            new_detection[alpha, 2] = np.random.uniform(low=IoU_threshold, high=1.0)
        elif mode == 3: # TP -> TP;
            update_num_gt = num_gt + 1
            alpha = np.random.choice(tp_index)
            new_detection = np.copy(detection)
            new_detection[alpha, 2] = np.random.uniform(low=new_detection[alpha, 2], high=1.0)
        elif mode == 6: # TP -> FP;
            update_num_gt = num_gt - 1
            alpha = np.random.choice(tp_index)
            new_detection = np.copy(detection)
            new_detection[alpha, 0] = 0
            new_detection[alpha, 2] = np.random.uniform(low=0, high=IoU_threshold)
        elif mode == 7: # TP -> TP;
            update_num_gt = num_gt - 1
            alpha = np.random.choice(tp_index)
            new_detection = np.copy(detection)
            new_detection[alpha, 2] = np.random.uniform(low=IoU_threshold, high=new_detection[alpha, 2])

        precision, recall = get_precision_recall(new_detection, update_num_gt)
        new_ap_11 = get_AP_11(precision, recall)
        new_ap_101 = get_AP_101(precision, recall)
        new_ap_all = get_AP_all(precision, recall)
        new_lrp = get_LRP(new_detection, update_num_gt, IoU_threshold)

        info_ap_11 = np.zeros(3)
        info_ap_11[0] = detection[alpha][1]
        info_ap_11[1] = detection[alpha][2]
        info_ap_11[2] = new_ap_11 - ap_11

        info_ap_101 = np.zeros(3)
        info_ap_101[0] = detection[alpha][1]
        info_ap_101[1] = detection[alpha][2]
        info_ap_101[2] = new_ap_101 - ap_101

        info_ap_all = np.zeros(3)
        info_ap_all[0] = detection[alpha][1]
        info_ap_all[1] = detection[alpha][2]
        info_ap_all[2] = new_ap_all - ap_all

        info_lrp = np.zeros(3)
        info_lrp[0] = detection[alpha][1]
        info_lrp[1] = detection[alpha][2]
        info_lrp[2] = new_lrp - lrp

        ap_11_change.append(info_ap_11)
        ap_101_change.append(info_ap_101)
        ap_all_change.append(info_ap_all)
        lrp_change.append(info_lrp)

    ap_11_change = np.array(ap_11_change)
    ap_101_change = np.array(ap_101_change)
    ap_all_change = np.array(ap_all_change)
    lrp_change = np.array(lrp_change)

    # save results
    np.save(output_path + str(mode) + '_ap_11_change.npy', ap_11_change)
    np.save(output_path + str(mode) + '_ap_101_change.npy', ap_101_change)
    np.save(output_path + str(mode) + '_ap_all_change.npy', ap_all_change)
    np.save(output_path + str(mode) + '_lrp_change.npy', lrp_change)

if __name__ == '__main__':
    # set seed
    seed = 1000
    np.random.seed(seed)

    # set options
    num_detections = 2000
    min_confidence = 0.001
    IoU_threshold = 0.5
    num_gt = 1000
    num_tp = 800
    num_trials = 10000

    # run simulation
    parser = argparse.ArgumentParser()
    parser.add_argument('-OPTION', type=str, default='localization') # ['localization']
    opt = parser.parse_args()

    simulation_option = opt.OPTION

    if simulation_option == 'localization':
        output_path = './localization/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        simulate_localization(num_trials, num_detections, num_tp, num_gt, min_confidence, IoU_threshold, output_path, mode=2)
        simulate_localization(num_trials, num_detections, num_tp, num_gt, min_confidence, IoU_threshold, output_path, mode=3)
        simulate_localization(num_trials, num_detections, num_tp, num_gt, min_confidence, IoU_threshold, output_path, mode=5)
        simulate_localization(num_trials, num_detections, num_tp, num_gt, min_confidence, IoU_threshold, output_path, mode=6)
    elif simulation_option == 'existence':
        output_path = './existence/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        simulate_existence(num_trials, num_detections, num_tp, num_gt, min_confidence, IoU_threshold, output_path, mode=2)
        simulate_existence(num_trials, num_detections, num_tp, num_gt, min_confidence, IoU_threshold, output_path, mode=3)
        simulate_existence(num_trials, num_detections, num_tp, num_gt, min_confidence, IoU_threshold, output_path, mode=6)
        simulate_existence(num_trials, num_detections, num_tp, num_gt, min_confidence, IoU_threshold, output_path, mode=7)

