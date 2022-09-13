# Improving EczemaNet: Evaluation Methods and New Datasets

Final report can be found [here](https://imperiallondon.sharepoint.com/:f:/r/sites/365-tanakagroup/Shared%20Documents/AD%20ML%20Project/SWET%20DATA%20-%20Hywel%20Williams/IMAGE%20PROJECT/2022%20Ryan%20EczemaNet/Final%20report?csf=1&web=1&e=urMF5W)

## Environment Setup
The setup instructions can be found in Zihao Wang's [Github page](https://github.ic.ac.uk/tanaka-group/EczemaNet-DeepLearning-Segmentation)

## Dataset
Due to data privacy reasons, the dataset for this project cannot be unloaded. You can find the whole dataset on Tanaka Group's RDS directory with relevant permission.

## Directories:
Please change all relevant directories of data to your own data directories before running any of the code.

## Structure:
```bash
.
├── data                         # data files
├── environment.yml              # For environment setup
├── job_scripts                  # For job submissions to RCS
├── notebooks                    # Files in jupyter notebook. These are mainly intermediate files for e.g. report graphs
├── output_trained_on_aug_S+T    # Output files based on model trained on augmented S+T dataset
├── output_trained_on_aug_SWET   # Output files based on model trained on augmented SWET dataset
├── output_trained_on_aug_TLA    # Output files based on model trained on augmented TLA dataset
├── output_trained_on_S+T        # Output files based on model trained on S+T dataset
├── output_trained_on_SWET       # Output files based on model trained on SWET dataset
├── output_trained_on_TLA        # Output files based on model trained on TLA dataset
├── report_plotting              # Plotting files for the final report 
└── Mod_Log                      # This pdf file contains all changes made compared to Zihao Wang's code
```
## Training
The file for training models is `/src/train_batch2.py`, where you should have three input parameters from the command line:
* The type of segmentation (either `skin`, `SKIN`, `ad` or `AD`).
* The directory of the training set
* The preferred prefix for your model

An example of running the code could be found in `/job_scripts/ryan_new_job_script_SWET.pbs`. The submission of jobs should follow the format `qsub ABC.pbs`, where the script's name should replace ABC. The job size instructions for HPC can be found in [RCS pages](https://wiki.imperial.ac.uk/display/HPC/New+Job+sizing+guidance). Further instructions of using HPC can be found [here](https://wiki.imperial.ac.uk/display/HPC/High+Performance+Computing).

## Evaluation
Three types of evaluations for the segmentation model contained in two files:
* `eval_of_ad_identification.py`: evaluates the cropping quality of AD region or the cropping quality of skin region
* `eval_of_robustness.py`: evaluates the robustness


### For further details or questions, please contant Yingrui Ma: ryanma0427@gmail.com
