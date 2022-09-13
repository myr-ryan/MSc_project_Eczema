import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
import os
import matplotlib as mpl

dataset_type = 'SWET'

def plot_pie_chart(dataset_type):

    if dataset_type == 'SWET':
        csv_path = '/Users/ryanma/Desktop/Onedrive/OneDrive - Imperial College London/IC_spring/Individual Project/coding/ethnic_sorted_SWET.csv'
        SWET_image_path = '/Users/ryanma/Desktop/Onedrive/OneDrive - Imperial College London/IC_spring/Individual Project/Data/Original_datasets/SWET'
        SWET_train_path = '/Users/ryanma/Desktop/Onedrive/OneDrive - Imperial College London/IC_spring/Individual Project/Data/Segmentation_training_data/dataset_SWET/training_set_corrected_SWET/reals'
        SWET_val_path = '/Users/ryanma/Desktop/Onedrive/OneDrive - Imperial College London/IC_spring/Individual Project/Data/Segmentation_training_data/dataset_SWET/validation_set_corrected_SWET/reals'
        SWET_test_path = '/Users/ryanma/Desktop/Onedrive/OneDrive - Imperial College London/IC_spring/Individual Project/Data/Segmentation_training_data/dataset_SWET/test_set_SWET/reals'
        SWET_real_files = os.listdir(SWET_train_path) + os.listdir(SWET_test_path) + os.listdir(SWET_val_path)
        # print(SWET_real_files)
        eth_id = pd.read_csv(csv_path)['refno'].values.tolist()
        eth = pd.read_csv(csv_path)['ethnic'].values.tolist()

        counted = {}

        for i in range(len(eth_id)):
        # for i in range(1):
            file_dir = os.path.join(SWET_image_path, '0' + str(eth_id[i]))
            if os.path.exists(file_dir):
                for j in os.listdir(file_dir):
                    # for k in SWET_real_files:
                    # print(str('0' + str(eth_id[i]) + '_' + j))
                    if str('0' + str(eth_id[i]) + '_' + j) in SWET_real_files:
                        if str(eth[i]) in counted.keys():
                            counted[eth[i]] += 1
                        else:
                            counted[eth[i]] = 1

                # num_of_img = len(os.listdir(file_dir))
                # if str(eth[i]) in counted.keys():
                #     counted[eth[i]] += num_of_img
                # else:
                #     counted[eth[i]] = num_of_img

        # print(counted)

        
        # All capital
        counted['Unknown'] = counted.pop('not stated/unknown')
        counted['Bangladeshi'] = counted.pop('bangladeshi')
        counted['Chinese'] = counted.pop('chinese')
        counted['Black-African'] = counted.pop('black-african')
        counted['Black-Caribbean'] = counted.pop('black-caribbean')
        
        counted['Black-Other'] = counted.pop('black-other')
        counted['Indian'] = counted.pop('indian')
        counted['Mixed'] = counted.pop('mixed')
        counted['Other'] = counted.pop('other')
        counted['White'] = counted.pop('white')
        counted['Pakistani'] = counted.pop('pakistani')


        # # There is one patient in 'Other' that do not wish to state the ethinicity
        # counted['Other'] = counted['Other'] - 1
        # # Not stated/unknown
        # counted.pop('not stated/unknown')

        # Plot pie chart
        labels = counted.keys()
        print(counted)
        explode = (0,0,0,0,0,0,0,0,0,0.1,0)
        mpl.rcParams['font.size'] = 25.0
        colors = ['r', 'y', 'Orange', 'm', 'c', 'g', 'LimeGreen', 'LightPink', 'Beige', 'DodgerBlue', 'MediumPurple']
        # colors = ['#25B7BE', '#EE756E', '#008000', '#0B66B0', '#00A9BD', '#40B58E', '#EFE342', '#BB3B1F', '#B31E22', '#1E2C6C', '#74151B']
        # colors = ['#25B7BE', '#EE756E', '#008000', '#1E2C6C', '#0B66B0', '#00A9BD', '#40B58E', '#EFE342', '#BB3B1F', '#B31E22', '#D52527']

        plt.pie(counted.values(),  colors=colors, explode=explode, autopct='%1.1f%%', pctdistance=1.15, labeldistance=1.2)
        # labels=labels,
         # autopct='%1.1f%%'
        # , textprops={'fontsize': 14}
        # textprops={'weight':'bold'}
        plt.legend(labels, loc='upper right', fontsize=20)
        
        plt.show()

        return counted

    elif dataset_type == 'TLA':
        csv_path = '/Users/ryanma/Desktop/Onedrive/OneDrive - Imperial College London/IC_spring/Individual Project/Data/Original_datasets/mergedFile.csv'
        # eth = pd.read_csv(csv_path)['ethnicity'].values.tolist()
        eth_id = pd.read_csv(csv_path)['id'].values.tolist()
        eth_id = list(set(eth_id))
        eth = pd.read_csv(csv_path)['ethnicity']

        # Count each ethnicity number, return type of dictionary
        counted = Counter(eth)
        # print(counted)

        # Not stated/unknown
        counted.pop('Not Stated/Unknown')

        counted['Unknown-white'] = 60
        counted['Unknown-Nonwhite'] = 184

        # Plot pie chart
        labels = counted.keys()
        print(counted)
        explode = (0,0,0,0,0,0,0.1,0)
        colors = ['r', 'y', 'Orange', 'm', 'c', 'g', 'DodgerBlue', 'MediumPurple']
        # colors = ['#1E2C6C', '#0B66B0', '#00A9BD', '#40B58E', '#EFE342', '#BB3B1F', '#B31E22', '#D52527']
        plt.pie(counted.values(), colors=colors, explode=explode, autopct='%1.1f%%')
        plt.legend(labels, loc='upper right', fontsize=20)
        plt.show()

        return counted


    else:
        print('Please check variable <dataset_type>')

        return {}


SWET_data = plot_pie_chart('SWET')
TLA_data = plot_pie_chart('TLA')

combined = SWET_data.copy()
combined.update(TLA_data)

combined['Bangladeshi'] += 14
combined['Pakistani'] += 19
combined['Black-Caribbean'] += 24
combined['Black-Other'] += 1
combined['Indian'] += 66


# num = combined.pop('Unknown-white')

combined['White'] += combined.pop('Unknown-white')


# Plot pie chart
labels = combined.keys()
print(combined)
explode = (0,0,0,0,0,0,0,0,0,0.1,0,0,0)
colors = ['r', 'y', 'Orange', 'm', 'c', 'g', 'Salmon', 'LightPink', 'DarkGray', 'DodgerBlue','LimeGreen','k', 'MediumPurple']
plt.pie(combined.values(), colors=colors, explode=explode, autopct='%1.1f%%', pctdistance=1.1, labeldistance=1.2)
plt.legend(labels, loc='upper right', fontsize=19)
plt.show()




