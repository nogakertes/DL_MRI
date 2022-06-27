"""configuration file for project"""

EXP_NAME = 'baseline_vanilla_Unet_data_consistency_MSE_Adam_8_in_ch_28Jun2022'
# EXP_NAME = 'Vanilla_Unet_26Jun2022_testSSIMreg_take1'
# CLEARML = False
# user = 'amitay'
# BATCH_SIZE = 8
# user = 'noga'
user = 'triton'
CLEARML = False
BATCH_SIZE = 32

DATA_CONSISTENCY = True
TEST_BATCH_SIZE = 1
EPOCHS = 50
LR = 1e-4
LR_PATIENCE = 5
LR_FACTOR = 0.9
INPUT_CHANNEL_SIZE = 16
SAVE_PLOTS = True
SAVE_NET = True
NOGA_DATA_PATH = 'C:/Users/Nogas/Desktop/fastmri_data/'
NOGA_EXP_PATH = 'C:/Users/Nogas/Desktop/Master/mriDL/experiments'
AMITAY_DATA_PATH = '/Users/amitaylev/PycharmProjects/DL_MRI/'
AMITAY_EXP_PATH = '/Users/amitaylev/Desktop/Amitay/Msc/2nd semester/DeepMRI/Accelerated Reconstruction/python_project/experiments'
TRITON_DATA_PATH = '/home/stu1/'
TRITON_EXP_PATH = '/home/stu1/experiments'
