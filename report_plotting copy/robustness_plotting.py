import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator, FormatStrFormatter
import pandas as pd
import os
import numpy as np


# PROJ_DIR = "/rds/general/user/ym521/home/EczemaNet-DeepLearning-Segmentation-master"
PROJ_DIR = "/Users/ryanma/Desktop/Onedrive/OneDrive - Imperial College London/IC_spring/Individual Project/Data/Segmentation_evaluations"
# OUTPUT_SWET_DIR = os.path.join(PROJ_DIR, 'output')

pert_methods = ['Blur', 'Brightness', 'Contrast', 'DefocusBlur', 'ElasticTransform', 'GaussianNoise', 'ImpulseNoise', 'MotionBlur', 'Pixelate', 'ShotNoise', 'ZoomBlur']

# pert_methods = ['Blur', 'Brightness', 'Contrast', 'DefocusBlur', 'GaussianNoise', 'ImpulseNoise', 'MotionBlur', 'Pixelate', 'ShotNoise', 'ZoomBlur']


output_names = ['SWET', 'TLA', 'S+T', 'aug_SWET', 'aug_TLA', 'aug_S+T']




def read_csvfiles(output_dir, pert_methods, severity_level):
    """
        This function is to read the robustness evaluation files from an output directory
   
        # Argument(s):
            output_dir: the output directory for either trained on SWET, TLA, S+T, aug SWET, aug TLA, or aug S+T.
            pert_methods: the perturbation method suffix
            severity_level: the severity level of that perturbation method, from 1 to 3

        # Return(s):
            SWET: the result csv file that was tested on SWET dataset
            TLA: the result csv file that was tested on TLA dataset
            S+T: the result csv file that was tested on S+T dataset

    """

    output = os.path.join(PROJ_DIR, output_dir)
    rob = pd.read_csv(os.path.join(output, 'robustness_evaluation_skin_severity_' + str(severity_level) + '_' + pert_methods + '.csv'))
    degradation = rob.values[-1, 2:3][0]

    return degradation


# Degradation sum list, it contains 6 lists for 6 different models, each list is of length 11, correpsonding to 11 perturbation methods
deg_sum_list = []
for i in range(len(output_names)):
    deg_sum_list.append([])
# deg_sum_list = [[], [], [], [], [], []]

# First loop through each model
for i in range(len(output_names)):
    out_dir = 'output_trained_on_' + output_names[i] + '/skin_seg/robustness/new_method'
    # Then loop through each perturbation method
    for method in pert_methods:
        temp_deg_sum = 0
        # Then loop through the three severity levels
        for severity in range(1, 4):
            degradation = read_csvfiles(out_dir, method, severity)
            temp_deg_sum += degradation

        deg_sum_list[i].append(temp_deg_sum)

# transpose
# print(deg_sum_list)
deg_sum_list = list(map(list, zip(*deg_sum_list)))

# print(deg_sum_list)
# print('---')
# # print(deg_sum_list[0][1])
# print(deg_sum_list[0][3:4])


# # Reference model --- SWET, TLA, S+T
# y = ['SWET+DA', 'TLA+DA', 'S+T+DA']
# for i in range(len(pert_methods)):
#     # print(np.divide(deg_sum_list[i][1::], deg_sum_list[i][0]))
#     plt.plot(([np.divide(deg_sum_list[i][3:4], deg_sum_list[i][0]), np.divide(deg_sum_list[i][4:5], deg_sum_list[i][1]), np.divide(deg_sum_list[i][5:6], deg_sum_list[i][2])]), y)

# plt.legend(pert_methods)
# plt.show()

fig = plt.gcf()
fig.set_size_inches(12.5,3.5)
pert_methods = ['Blur', 'Brightness', 'Contrast', 'DefocusBlur', 'Elastic', 'GaussianNoise', 'ImpulseNoise', 'MotionBlur', 'Pixelate', 'ShotNoise', 'ZoomBlur']


# Change this line to get different plots
# The names are "Aug SWET (ref model SWET)"; "Aug TLA (ref model TLA)"; "Aug S+T (ref model S+T)"; "S+T (ref model SWET)", "Aug S+T (ref model Aug SWET)"

# PLOT_NAME = "Aug SWET (ref model SWET)"
# PLOT_NAME = "Aug TLA (ref model TLA)"
# PLOT_NAME = "Aug S+T (ref model S+T)"
# PLOT_NAME = "S+T (ref model SWET)"
# PLOT_NAME = "Aug S+T (ref model Aug SWET)"
PLOT_NAME = "Aug S+T (ref model Aug TLA)"

# For SWET and aug SWET
x_deg = []
if PLOT_NAME == "Aug SWET (ref model SWET)":
    for i in range(len(pert_methods)):
        x_deg.append(np.divide(deg_sum_list[i][3:4], deg_sum_list[i][0])[0])
elif PLOT_NAME == "Aug TLA (ref model TLA)":
    for i in range(len(pert_methods)):
        x_deg.append(np.divide(deg_sum_list[i][4:5], deg_sum_list[i][1])[0])
elif PLOT_NAME == "Aug S+T (ref model S+T)":
    for i in range(len(pert_methods)):
        x_deg.append(np.divide(deg_sum_list[i][5:6], deg_sum_list[i][2])[0])
elif PLOT_NAME == "S+T (ref model SWET)":
    for i in range(len(pert_methods)):
        x_deg.append(np.divide(deg_sum_list[i][2:3], deg_sum_list[i][0])[0])
elif PLOT_NAME == "Aug S+T (ref model Aug SWET)":
    for i in range(len(pert_methods)):
        x_deg.append(np.divide(deg_sum_list[i][5:6], deg_sum_list[i][3])[0])
elif PLOT_NAME == "Aug S+T (ref model Aug TLA)":
    for i in range(len(pert_methods)):
        x_deg.append(np.divide(deg_sum_list[i][5:6], deg_sum_list[i][4])[0])
else:
    print("Please check variable <PLOT_NAME>")

if x_deg != []:
    # x_deg.append(np.mean(x_deg))
    # figure(figsize=(4,1), dpi=200)
    # pert_methods.append('Mean')
    plt.barh(pert_methods, x_deg)
    plt.barh('Mean', np.mean(x_deg), color='red')
    print(x_deg)
    print(np.mean(x_deg))
    plt.title(PLOT_NAME, fontsize=14)
    plt.vlines(x=1.0, ymin=-0.5, ymax=len(pert_methods)+1.0, color='black', ls = ':', label='Improvement Threshold')
    plt.xlim([0.0, 1.2])
    plt.legend(fontsize=12, loc='lower right')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Save the figure
    # plt.savefig('robustness_' + PLOT_NAME + '.png', dpi=200)
    plt.show()


# # For TLA and aug TLA
# fig = plt.gcf()
# fig.set_size_inches(12,3.5)
# x_deg = []
# for i in range(len(pert_methods)):
#     x_deg.append(np.divide(deg_sum_list[i][4:5], deg_sum_list[i][1])[0])

# plt.barh(pert_methods, x_deg)
# plt.title('TLA+DA', fontsize=14)
# plt.vlines(x=1.0, ymin=-0.5, ymax=len(pert_methods)-0.5, color='black', ls = ':', label='Improvement Threshold')
# plt.xlim([0.0, 1.2])
# plt.legend(fontsize=14, loc='lower right')
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)

# # Save the figure
# plt.savefig('robustness_2.png', dpi=200)
# plt.show()

# # For S+T and aug S+T
# fig = plt.gcf()
# fig.set_size_inches(12,3.5)
# x_deg = []
# for i in range(len(pert_methods)):
#     x_deg.append(np.divide(deg_sum_list[i][5:6], deg_sum_list[i][2])[0])

# plt.barh(pert_methods, x_deg)
# plt.title('S+T+DA')
# plt.vlines(x=1.0, ymin=-0.5, ymax=len(pert_methods)-0.5, color='black', ls = ':', label='Improvement Threshold')
# plt.xlim([0.0, 1.2])
# plt.legend(fontsize=14, loc='lower right')
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)

# # Save the figure
# plt.savefig('robustness_3.png', dpi=200)
# plt.show()












