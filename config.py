"""configuration file for project"""

CLEARML = True
EXP_NAME = 'Vanilla_Unet_22Jun2022_SIGMOID_exp_scheduler'
# user = 'amitay'
# user = 'noga'
user = 'triton'
BATCH_SIZE = 32
TEST_BATCH_SIZE = 1
EPOCHS = 50
LR = 1e-4
LR_PATIENCE = 5
LR_FACTOR = 0.1
INPUT_CHANNEL_SIZE = 8
SAVE_PLOTS = True
SAVE_NET = True
NOGA_DATA_PATH = 'C:/Users/Nogas/Desktop/fastmri_data/'
NOGA_EXP_PATH = 'C:/Users/Nogas/Desktop/Master/mriDL/experiments'
AMITAY_DATA_PATH = '/Users/amitaylev/PycharmProjects/DL_MRI/'
AMITAY_EXP_PATH = '/Users/amitaylev/Desktop/Amitay/Msc/2nd semester/DeepMRI/Accelerated Reconstruction/python_project/experiments'
TRITON_DATA_PATH = '/home/stu1/'
TRITON_EXP_PATH = '/home/stu1/experiments'
