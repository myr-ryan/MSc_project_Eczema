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
    SWET = pd.read_csv(os.path.join(output, 'new_skin_on_SWET_fewer.csv'))
    TLA = pd.read_csv(os.path.join(output, 'new_skin_on_TLA.csv'))
    ST = pd.read_csv(os.path.join(output, 'new_skin_on_S+T_fewer.csv'))

    return SWET, TLA, ST


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


    # y = ['SWET','1', '2', 'S+T', 'SWET+DA', '3', 'S+T+DA', '4']
    y1 = np.linspace(0, 6, num=4)
    y2 = np.linspace(1, 7, num=4)

    # y = np.linspace(1, 3, 1)
    axs[plot_index].errorbar([x1[0], x1[2], x1[4], x1[6]], y1, ecolor='blue', fmt='ob', xerr=[xerr1[0], xerr1[2], xerr1[4], xerr1[6]])
    axs[plot_index].errorbar([x1[1], x1[3], x1[5], x1[7]], y2, ecolor='black', fmt='ok', xerr=[xerr1[1], xerr1[3], xerr1[5], xerr1[7]])

    # axs[plot_index].grid(visible=True)
    axs[plot_index].axis(xmin = 0.5, xmax = 1.0)
    axs[plot_index].axis(ymin = -0.5, ymax = 7.5)
    
    axs[plot_index].yaxis.set_major_locator(FixedLocator([1.5, 3.5, 5.5]))
    axs[plot_index].yaxis.grid(True, which='major')
    axs[plot_index].set_yticklabels(['', '', ''], minor=False)
    axs[plot_index].yaxis.set_minor_locator(FixedLocator([0.5, 2.5, 4.5, 6.5]))
    # axs[plot_index].set_yticks([0.5, 2.5, 4.5, 6.5], ['SWET', 'S+T', 'SWET+DA', 'S+T+DA'], minor=True)
    axs[plot_index].set_yticklabels(['SWET', 'S+T', 'SWET+DA', 'S+T+DA'], minor=True)
    axs[plot_index].tick_params(axis='both', labelsize=15, which='both')
    axs[plot_index].xaxis.grid(True)
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
    axs[plot_index].legend([evaluation_type + ' on SWET', evaluation_type + ' on S+T'], fontsize=15)
    # axs[plot_index].legend([evaluation_type + ' on S+T'], fontsize=15)


    if savefile:
        axs[plot_index].savefig(evaluation_type + '.png', bbox_inches='tight', dpi=200)
    # plt.show()



# Trained on SWET
SWET_on_SWET, SWET_on_TLA, SWET_on_ST = read_csvfiles('output_trained_on_SWET/skin_seg/cropping_quality/eval_skin_region')
# Trained on Aug SWET
Aug_SWET_on_SWET, Aug_SWET_on_TLA, Aug_SWET_on_ST = read_csvfiles('output_trained_on_aug_SWET/skin_seg/cropping_quality/eval_skin_region')
# Trained on TLA
TLA_on_SWET, TLA_on_TLA, TLA_on_ST  = read_csvfiles('output_trained_on_TLA/skin_seg/cropping_quality/eval_skin_region')
# Trained on Aug TLA
Aug_TLA_on_SWET, Aug_TLA_on_TLA, Aug_TLA_on_ST = read_csvfiles('output_trained_on_aug_TLA/skin_seg/cropping_quality/eval_skin_region')
# Trained on S+T
ST_on_SWET, ST_on_TLA, ST_on_ST = read_csvfiles('output_trained_on_S+T/skin_seg/cropping_quality/eval_skin_region')
# Trained on Aug S+T
Aug_ST_on_SWET, Aug_ST_on_TLA, Aug_ST_on_ST = read_csvfiles('output_trained_on_aug_S+T/skin_seg/cropping_quality/eval_skin_region')


"""
    Trained on SWET
"""
mean_SS, se_SS = get_data_from_csv(SWET_on_SWET)
mean_ST, se_ST = get_data_from_csv(SWET_on_TLA)
mean_SST, se_SST = get_data_from_csv(SWET_on_ST)


"""
    Trained on Aug SWET
"""
mean_aug_SS, se_aug_SS = get_data_from_csv(Aug_SWET_on_SWET)
mean_aug_ST, se_aug_ST = get_data_from_csv(Aug_SWET_on_TLA)
mean_aug_SST, se_aug_SST = get_data_from_csv(Aug_SWET_on_ST)

"""
    Trained on TLA
"""
mean_TS, se_TS = get_data_from_csv(TLA_on_SWET)
mean_TT, se_TT = get_data_from_csv(TLA_on_TLA)
mean_TST, se_TST = get_data_from_csv(TLA_on_ST)

"""
    Trained on Aug TLA
"""
mean_aug_TS, se_aug_TS = get_data_from_csv(Aug_TLA_on_SWET)
mean_aug_TT, se_aug_TT = get_data_from_csv(Aug_TLA_on_TLA)
mean_aug_TST, se_aug_TST = get_data_from_csv(Aug_TLA_on_ST)

"""
    Trained on S+T
"""
mean_STS, se_STS = get_data_from_csv(ST_on_SWET)
mean_STT, se_STT = get_data_from_csv(ST_on_TLA)
mean_STST, se_STST = get_data_from_csv(ST_on_ST)

"""
    Trained on Aug S+T
"""
mean_aug_STS, se_aug_STS = get_data_from_csv(Aug_ST_on_SWET)
mean_aug_STT, se_aug_STT = get_data_from_csv(Aug_ST_on_TLA)
mean_aug_STST, se_aug_STST = get_data_from_csv(Aug_ST_on_ST)



"""
    Recall plotting
"""
# Tested on SWET
# recall_x_S = [mean_SS[0], mean_ST[0], mean_SST[0], mean_TS[0], mean_TT[0], mean_TST[0], mean_STS[0], mean_STT[0], mean_STST[0], mean_aug_SS[0], mean_aug_ST[0], mean_aug_SST[0], mean_aug_TS[0], mean_aug_TT[0], mean_aug_TST[0], mean_aug_STS[0], mean_aug_STT[0], mean_aug_STST[0]]
recall_x_S = [mean_SS[0], mean_SST[0], mean_STS[0], mean_STST[0], mean_aug_SS[0], mean_aug_SST[0], mean_aug_STS[0], mean_aug_STST[0]]
print('Recall: ', recall_x_S)


# recall_xerr_S = [se_SS[0], se_ST[0], se_SST[0], se_TS[0], se_TT[0], se_TST[0], se_STS[0], se_STT[0], se_STST[0], se_aug_SS[0], se_aug_ST[0], se_aug_SST[0], se_aug_TS[0], se_aug_TT[0], se_aug_TST[0], se_aug_STS[0], se_aug_STT[0], se_aug_STST[0]]
recall_xerr_S = [se_SS[0], se_SST[0], se_STS[0], se_STST[0], se_aug_SS[0], se_aug_SST[0], se_aug_STS[0], se_aug_STST[0]]
print('Recall se: ', recall_xerr_S)



fig, axs = plt.subplots(4)
plot_index = 0


plot_errorbar(recall_x_S, recall_xerr_S, 'Recall', plot_index, savefile=False)
plot_index += 1


# """
#     Precision plotting
# """

# Tested on SWET
# precision_x_S = [mean_SS[1], mean_ST[1], mean_SST[1], mean_TS[1], mean_TT[1], mean_TST[1], mean_STS[1], mean_STT[1], mean_STST[1], mean_aug_SS[1], mean_aug_ST[1], mean_aug_SST[1], mean_aug_TS[1], mean_aug_TT[1], mean_aug_TST[1], mean_aug_STS[1], mean_aug_STT[1], mean_aug_STST[1]]
precision_x_S = [mean_SS[1], mean_SST[1], mean_STS[1], mean_STST[1], mean_aug_SS[1], mean_aug_SST[1], mean_aug_STS[1], mean_aug_STST[1]]
# print('Precision: ', precision_x_S)

# precision_xerr_S = [se_SS[1], se_ST[1], se_SST[1], se_TS[1], se_TT[1], se_TST[1], se_STS[1], se_STT[1], se_STST[1], se_aug_SS[1], se_aug_ST[1], se_aug_SST[1], se_aug_TS[1], se_aug_TT[1], se_aug_TST[1], se_aug_STS[1], se_aug_STT[1], se_aug_STST[1]]
precision_xerr_S = [se_SS[1], se_SST[1], se_STS[1], se_STST[1], se_aug_SS[1], se_aug_SST[1], se_aug_STS[1], se_aug_STST[1]]
# print('Precision se: ', precision_xerr_S)



plot_errorbar(precision_x_S, precision_xerr_S, 'Precision', plot_index, savefile=False)
plot_index += 1





# """
#     F1 score plotting
# """

# Tested on SWET
# f1_x_S = [mean_SS[2], mean_ST[2], mean_SST[2], mean_TS[2], mean_TT[2], mean_TST[2], mean_STS[2], mean_STT[2], mean_STST[2], mean_aug_SS[2], mean_aug_ST[2], mean_aug_SST[2], mean_aug_TS[2], mean_aug_TT[2], mean_aug_TST[2], mean_aug_STS[2], mean_aug_STT[2], mean_aug_STST[2]]
f1_x_S = [mean_SS[2], mean_SST[2], mean_STS[2], mean_STST[2], mean_aug_SS[2], mean_aug_SST[2], mean_aug_STS[2], mean_aug_STST[2]]

# print('F1: ', f1_x_S)
# f1_xerr_S = [se_SS[2], se_ST[2], se_SST[2], se_TS[2], se_TT[2], se_TST[2], se_STS[2], se_STT[2], se_STST[2], se_aug_SS[2], se_aug_ST[2], se_aug_SST[2], se_aug_TS[2], se_aug_TT[2], se_aug_TST[2], se_aug_STS[2], se_aug_STT[2], se_aug_STST[2]]
f1_xerr_S = [se_SS[2], se_SST[2], se_STS[2], se_STST[2], se_aug_SS[2], se_aug_SST[2], se_aug_STS[2], se_aug_STST[2]]
# print('F1 se: ', f1_xerr_S)


plot_errorbar(f1_x_S, f1_xerr_S, 'F1-score', plot_index, savefile=False)
plot_index += 1


# """
#     Accuracy plotting
# """

# acc_x_S = [mean_SS[3], mean_ST[3], mean_SST[3], mean_TS[3], mean_TT[3], mean_TST[3], mean_STS[3], mean_STT[3], mean_STST[3], mean_aug_SS[3], mean_aug_ST[3], mean_aug_SST[3], mean_aug_TS[3], mean_aug_TT[3], mean_aug_TST[3], mean_aug_STS[3], mean_aug_STT[3], mean_aug_STST[3]]
acc_x_S = [mean_SS[3], mean_SST[3], mean_STS[3], mean_STST[3], mean_aug_SS[3], mean_aug_SST[3], mean_aug_STS[3], mean_aug_STST[3]]

# print('Accuracy: ', acc_x_S)
# acc_xerr_S = [se_SS[3], se_ST[3], se_SST[3], se_TS[3], se_TT[3], se_TST[3], se_STS[3], se_STT[3], se_STST[3], se_aug_SS[3], se_aug_ST[3], se_aug_SST[3], se_aug_TS[3], se_aug_TT[3], se_aug_TST[3], se_aug_STS[3], se_aug_STT[3], se_aug_STST[3]]
acc_xerr_S = [se_SS[3], se_SST[3], se_STS[3], se_STST[3], se_aug_SS[3], se_aug_SST[3], se_aug_STS[3], se_aug_STST[3]]

# print('Accuracy se: ', acc_xerr_S)


plot_errorbar(acc_x_S, acc_xerr_S, 'Accuracy', plot_index, savefile=False)
plot_index += 1


plt.show()




