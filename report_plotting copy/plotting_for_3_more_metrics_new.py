import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator, FormatStrFormatter
import pandas as pd
import os
import numpy as np


# PROJ_DIR = "/rds/general/user/ym521/home/EczemaNet-DeepLearning-Segmentation-master"
PROJ_DIR = "/Users/ryanma/Desktop/Onedrive/OneDrive - Imperial College London/IC_spring/Individual Project/Data/Segmentation_evaluations"
# OUTPUT_SWET_DIR = os.path.join(PROJ_DIR, 'output')

def read_csvfiles(output_dir):
    """
        This function is to read three csv files (SWET, TLA, S+T) from an output directory.
   
        # Argument(s):
            output_dir: the output directory for either trained on SWET, TLA or S+T

        # Return(s):
            SWET: the result csv file that was tested on SWET dataset
            TLA: the result csv file that was tested on TLA dataset
            S+T: the result csv file that was tested on S+T dataset

    """

    output = os.path.join(PROJ_DIR, output_dir)
    SWET = pd.read_csv(os.path.join(output, 'new_skin_on_SWET_more_metrics.csv'))

    return SWET


def get_data_from_csv(csv_name):
    """
        This function is to get mean and se from each csv file, which are located in the last two rows of the csv files
    """
    mean = csv_name.values[-2, 1:5]
    se = csv_name.values[-1, 1:5]

    return mean, se



def plot_errorbar(x1, xerr1, evaluation_type, plot_index, savefile=False):
    """
        This function is for plotting results in the same graph
    """


    y = ['SWET', 'TLA', 'S+T', 'SWET+DA', 'TLA+DA', 'S+T+DA']
    # y = np.linspace(1, 3, 1)
    axs[plot_index].errorbar(x1, y, ecolor='black', fmt='ok', xerr=xerr1)
    axs[plot_index].grid(visible=True)
    axs[plot_index].axis(xmin = 0.5, xmax = 1.0)
    axs[plot_index].axis(ymin = -1, ymax = 6)
    axs[plot_index].tick_params(axis='both', labelsize=15)
    # axs[plot_index].get_xaxis().set_ticks(fontsize=18)
    # axs[plot_index].set_xlabel('', fontsize=18)
    # axs[0].xlim([0.5, 1.0])
    # axs[0].ylim([-1, 6])
    # axs[0].legend(['Test on SWET'], fontsize=18)
    # axs[0].xlabel(evaluation_type, fontsize=18)
    # axs[0].ylabel('Trained on', fontsize=18)
    # axs[0].xticks(fontsize=18)
    # axs[0].yticks(fontsize=18)
    # axs[0].rcParams.update({'font.size': 18})
    # plt.title(evaluation_type)
    # plt.show()
    axs[plot_index].legend([evaluation_type], fontsize=15)

    if savefile:
        axs[plot_index].savefig(evaluation_type + '.png', bbox_inches='tight', dpi=200)
    # plt.show()



# Trained on SWET
SWET_on_SWET = read_csvfiles('output_trained_on_SWET/skin_seg/cropping_quality/eval_AD_region')
# Trained on Aug SWET
Aug_SWET_on_SWET = read_csvfiles('output_trained_on_aug_SWET/skin_seg/cropping_quality/eval_AD_region')
# Trained on TLA
TLA_on_SWET = read_csvfiles('output_trained_on_TLA/skin_seg/cropping_quality/eval_AD_region')
# Trained on Aug TLA
Aug_TLA_on_SWET = read_csvfiles('output_trained_on_aug_TLA/skin_seg/cropping_quality/eval_AD_region')
# Trained on S+T
ST_on_SWET = read_csvfiles('output_trained_on_S+T/skin_seg/cropping_quality/eval_AD_region')
# Trained on Aug S+T
Aug_ST_on_SWET = read_csvfiles('output_trained_on_aug_S+T/skin_seg/cropping_quality/eval_AD_region')


"""
    Trained on SWET
"""
mean_SS, se_SS = get_data_from_csv(SWET_on_SWET)


"""
    Trained on Aug SWET
"""
mean_aug_SS, se_aug_SS = get_data_from_csv(Aug_SWET_on_SWET)


"""
    Trained on TLA
"""
mean_TS, se_TS = get_data_from_csv(TLA_on_SWET)

"""
    Trained on Aug TLA
"""
mean_aug_TS, se_aug_TS = get_data_from_csv(Aug_TLA_on_SWET)

"""
    Trained on S+T
"""
mean_STS, se_STS = get_data_from_csv(ST_on_SWET)

"""
    Trained on Aug S+T
"""
mean_aug_STS, se_aug_STS = get_data_from_csv(Aug_ST_on_SWET)



"""
    Recall plotting
"""
# Tested on SWET
recall_x_S = [mean_SS[0], mean_TS[0], mean_STS[0], mean_aug_SS[0], mean_aug_TS[0], mean_aug_STS[0]]
print('Recall: ', recall_x_S)
recall_xerr_S = [se_SS[0],  se_TS[0], se_STS[0], se_aug_SS[0], se_aug_TS[0], se_aug_STS[0]]
print('Recall se: ', recall_xerr_S)

fig, axs = plt.subplots(2)
plot_index = 0


# plot_errorbar(recall_x_S, recall_xerr_S, 'Recall', plot_index, savefile=False)
# plot_index += 1


# """
#     Precision plotting
# """

# Tested on SWET
precision_x_S = [mean_SS[1], mean_TS[1], mean_STS[1], mean_aug_SS[1], mean_aug_TS[1], mean_aug_STS[1]]
print('Precision: ', precision_x_S)
precision_xerr_S = [se_SS[1], se_TS[1], se_STS[1], se_aug_SS[1], se_aug_TS[1], se_aug_STS[1]]
print('Precision se: ', precision_xerr_S)



# plot_errorbar(precision_x_S, precision_xerr_S, 'Precision', plot_index, savefile=False)
# plot_index += 1





# """
#     F1 score plotting
# """

# Tested on SWET
f1_x_S = [mean_SS[2], mean_TS[2], mean_STS[2], mean_aug_SS[2], mean_aug_TS[2], mean_aug_STS[2]]
print('F1: ', f1_x_S)
f1_xerr_S = [se_SS[2], se_TS[2], se_STS[2], se_aug_SS[2], se_aug_TS[2], se_aug_STS[2]]
print('F1 se: ', f1_xerr_S)


plot_errorbar(f1_x_S, f1_xerr_S, 'F1-score', plot_index, savefile=False)
plot_index += 1


# """
#     Accuracy plotting
# """

acc_x_S = [mean_SS[3], mean_TS[3], mean_STS[3], mean_aug_SS[3], mean_aug_TS[3], mean_aug_STS[3]]
print('Accuracy: ', acc_x_S)
acc_xerr_S = [se_SS[3], se_TS[3], se_STS[3], se_aug_SS[3], se_aug_TS[3], se_aug_STS[3]]
print('Accuracy se: ', acc_xerr_S)


plot_errorbar(acc_x_S, acc_xerr_S, 'Accuracy', plot_index, savefile=False)
plot_index += 1


plt.show()




