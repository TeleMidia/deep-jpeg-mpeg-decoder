import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import os
import glob
import lib.jpeg as jpg
from skimage.measure import compare_ssim, compare_psnr, compare_nrmse

exp_chart_folder = None
model_weights_folder1 = None
model_weights_folder2 = None
dict_chart_data = None
CONST_GAMA = 0.01
LAST_EPOCH = -1
BEST_VALIDATION_EPOCH = 0

class CustomMetric:
    def __init__(self):
        self.buffer_psnr = []
        self.buffer_ssim = []
        self.buffer_nrmse = []
       
    def feed(self, batch_y, predictions):
        batch_size = predictions.shape[0]
        for index in range(0, batch_size):
            self.buffer_psnr = np.concatenate((self.buffer_psnr, 
            compare_psnr(batch_y[index], predictions[index], data_range=255)), axis=None) 
            self.buffer_ssim = np.concatenate((self.buffer_ssim, 
            compare_ssim(batch_y[index], predictions[index], multichannel=True)), axis=None)
            self.buffer_nrmse = np.concatenate((self.buffer_nrmse, 
            compare_nrmse(batch_y[index], predictions[index])), axis=None)
            
    def result(self):
        return np.mean(self.buffer_psnr), np.mean(self.buffer_ssim), np.mean(self.buffer_nrmse)

    def reset_states(self):
        self.buffer_psnr = []
        self.buffer_ssim = []
        self.buffer_nrmse = []


def check_experiment_folders():
    global exp_chart_folder, model_weights_folder1,  model_weights_folder2
    if exp_chart_folder is None or model_weights_folder1 is None or model_weights_folder2 is None:
        return False
    return True

def create_experiment_folders(exp_id):
    global exp_chart_folder, model_weights_folder1, model_weights_folder2
    exp_chart_folder = os.path.join("model_save", exp_id, "chart_data")
    model_weights_folder1 = os.path.join("model_save", exp_id, "model_last_epoch")
    model_weights_folder2 = os.path.join("model_save", exp_id, "model_best_valid")
    if not os.path.exists(exp_chart_folder):
        os.makedirs(exp_chart_folder)
    if not os.path.exists(model_weights_folder1):
        os.makedirs(model_weights_folder1)
    if not os.path.exists(model_weights_folder2):
        os.makedirs(model_weights_folder2)    
    return 

def get_exp_folder_last_epoch():
    return os.path.join(model_weights_folder1, "model")
    
def get_exp_folder_best_valid():
    return os.path.join(model_weights_folder2, "model")

def load_experiment_data():
    assert check_experiment_folders()
    global exp_chart_folder, dict_chart_data, LAST_EPOCH
    path =  os.path.join(exp_chart_folder, "data.txt")
    if os.path.exists(path):
        with open(path, "r") as file:
            dict_chart_data = eval(file.readline())
            #print(dict_chart_data)
            print(dict_chart_data["epoch"])
            if len(dict_chart_data["epoch"]) > 0:
                LAST_EPOCH = int(dict_chart_data["epoch"][-1])
                #print(LAST_EPOCH)    
    else:
        dict_chart_data = {}
        dict_chart_data["epoch"] = []
        dict_chart_data["Train_MSE"] = []
        dict_chart_data["Valid_MSE"] = []
        dict_chart_data["PSNR"] = []
        dict_chart_data["SSIM"] = []
        dict_chart_data["NRMSE"] = []
        dict_chart_data["Best_Validation_Result"] = 0
        dict_chart_data["Best_Validation_Epoch"] = 0
    return

def get_model_last_data(mode="LastEpoch"):
    global LAST_EPOCHccccccccccccccccccccccccccc
    if mode =="LastEpoch":
        return LAST_EPOCH+1, dict_chart_data["Best_Validation_Result"]
    else: 
        return dict_chart_data["Best_Validation_Epoch"], dict_chart_data["Best_Validation_Result"]


def update_chart_data(epoch, train_mse, valid_mse, psnr, ssim, nrmse):
    assert check_experiment_folders()
    global exp_chart_folder,dict_chart_data
    assert dict_chart_data is not None
    path =  os.path.join(exp_chart_folder, "data.txt")

    if ssim > dict_chart_data["Best_Validation_Result"]:
        dict_chart_data["Best_Validation_Result"] = ssim
        dict_chart_data["Best_Validation_Epoch"] = epoch   

    dict_chart_data["epoch"].append(epoch)
    dict_chart_data["Train_MSE"].append(train_mse)
    dict_chart_data["Valid_MSE"].append(valid_mse)
    dict_chart_data["PSNR"].append(psnr)
    dict_chart_data["SSIM"].append(ssim)
    dict_chart_data["NRMSE"].append(nrmse)

    if os.path.exists(path):
        os.remove(path) 
    with open(path, "w") as file:
        file.write(str(dict_chart_data))
        
    return 

def annot_max(ax, x,y, op="min"):

    if op=="min":
        xmax = x[np.argmin(y)]
        ymax = y.min()
    else:
        xmax = x[np.argmax(y)]
        ymax = y.max()

    text= "epoch={}, result={:.6f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=1)
    arrowprops=dict(arrowstyle="->")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

    
def draw_chart():
    global dict_chart_data

    if len(dict_chart_data["epoch"]) == 0:
        return

    fig, axs = plt.subplots(4, figsize=(15,15))
  
    axs[0].plot(dict_chart_data["epoch"], dict_chart_data["Train_MSE"], linewidth=2, color="orange", label="Train_MSE")
    axs[0].plot(dict_chart_data["epoch"], dict_chart_data["Valid_MSE"], linewidth=2, color="blue", label="Valid_MSE")
    axs[0].legend(frameon=False, loc='upper center', ncol=2)
    #annot_max(axs[0], np.asarray(dict_chart_data["epoch"]), np.asarray(dict_chart_data["Valid_MSE"]) )

    axs[1].plot(dict_chart_data["epoch"], dict_chart_data["PSNR"], linewidth=2, color="green", label="PSNR")
    axs[1].legend(frameon=False, loc='upper center', ncol=1)
    annot_max(axs[1], np.asarray(dict_chart_data["epoch"]), np.asarray(dict_chart_data["PSNR"]), op="max")

    axs[2].plot(dict_chart_data["epoch"], dict_chart_data["SSIM"], linewidth=2, color="red", label="SSIM")
    axs[2].legend(frameon=False, loc='upper center', ncol=1)
    annot_max(axs[2], np.asarray(dict_chart_data["epoch"]), np.asarray(dict_chart_data["SSIM"]), op="max")

    axs[3].plot(dict_chart_data["epoch"], dict_chart_data["NRMSE"], linewidth=2, color="magenta", label="NRMSE")
    axs[3].legend(frameon=False, loc='upper center', ncol=1)
    annot_max(axs[3], np.asarray(dict_chart_data["epoch"]), np.asarray(dict_chart_data["NRMSE"]))

    plt.show()



def load_dataset(root_folder, limit=None):

    dataset_x = []
    dataset_y = []

    counter = 0

    qtable_luma_50, qtable_chroma_50 = jpg.generate_qtables(quality_factor=50)
    qtable_luma_10, qtable_chroma_10 = jpg.generate_qtables(quality_factor=10)

    for file_ in glob.iglob(root_folder+"/*/*.jpg"):
        img = open_image(file_)

        if img is None:
            print("Corrupted dataset!")
            return None, None

        dataset_x.append(jpg.encode_image(img, qtable_luma_10, qtable_chroma_10))
        dataset_y.append(jpg.encode_image(img, qtable_luma_50, qtable_chroma_50))

        counter += 1
        if limit != None and counter >= limit:
            break
        
    return np.array(dataset_x), np.array(dataset_y)

def show_samples(dataset_x, dataset_y, begin=0, end=1):

    qtable_luma_50, qtable_chroma_50 = jpg.generate_qtables(quality_factor=50)
    qtable_luma_10, qtable_chroma_10 = jpg.generate_qtables(quality_factor=10)

    quant = end - begin
    fig, axs = plt.subplots(quant, figsize=(15,15), frameon=True)
    for index in range(begin,end):
        axs[index].axis('off')
        img_x = jpg.decode_image(dataset_x[index].copy(), qtable_luma_10, qtable_chroma_10)
        img_y = jpg.decode_image(dataset_y[index].copy(), qtable_luma_50, qtable_chroma_50)
        axs[index].imshow(np.concatenate((cvt_bgr2rgb(img_x), cvt_bgr2rgb(img_y)), axis=1), vmin=0, vmax=255)
        
    plt.show()

def convert_batch_dct2rgb(dataset_x, dataset_y, predict):
    qtable_luma_50, qtable_chroma_50 = jpg.generate_qtables(quality_factor=50)
    qtable_luma_10, qtable_chroma_10 = jpg.generate_qtables(quality_factor=10)
    quant = predict.shape[0]
    list_dataset_x = []
    list_dataset_y = []
    list_predict = []
    for index in range(quant):
        list_dataset_x.append(cvt_bgr2rgb( jpg.decode_image(dataset_x[index].copy(), qtable_luma_10, qtable_chroma_10)))
        list_dataset_y.append(cvt_bgr2rgb( jpg.decode_image(dataset_y[index].copy(), qtable_luma_50, qtable_chroma_50)))
        list_predict.append(cvt_bgr2rgb( jpg.decode_image(predict[index].copy(), qtable_luma_50, qtable_chroma_50)))    

    return np.array(list_dataset_x), np.array(list_dataset_y), np.array(list_predict) 

def preview_sample(dataset_x, dataset_y, predict):
    fig = plt.imshow(np.concatenate((dataset_x, dataset_y, predict), axis=1), vmin=0, vmax=255)
    plt.axis('off')    
    return


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def open_image(path):
    img = cv2.imread(path)
    if img.shape[0]%8 != 0 or img.shape[1]%8 != 0:
        print("Invalid image size. Input image width and height should be multiple of 8.")
        exit()

    if img is None:
        print("Invalid image path.")
        exit()

    return img

def cvt_bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_shift_scale_maxmin(train_x, train_y, valid_x, valid_y):
    
    SHIFT_VALUE_X = 0
    SHIFT_VALUE_Y = 0
    SCALE_VALUE_X = 0
    SCALE_VALUE_Y = 0

    if np.amin(valid_x) < np.amin(train_x):
        SHIFT_VALUE_X = np.amin(valid_x)
    else:
        SHIFT_VALUE_X = np.amin(train_x)

    if np.amin(valid_y) < np.amin(train_y):
        SHIFT_VALUE_Y = np.amin(valid_y)
    else:
        SHIFT_VALUE_Y = np.amin(train_y)

    if np.amax(valid_x) > np.amax(train_x):
        SCALE_VALUE_X = np.amax(valid_x)
    else:
        SCALE_VALUE_X = np.amax(train_x)

    if np.amax(valid_y) > np.amax(train_y):
        SCALE_VALUE_Y = np.amax(valid_y)
    else:
        SCALE_VALUE_Y = np.amax(train_y)

    
    SHIFT_VALUE_X = SHIFT_VALUE_X*-1
    SHIFT_VALUE_Y = SHIFT_VALUE_Y*-1
    SCALE_VALUE_X += SHIFT_VALUE_X
    SCALE_VALUE_Y += SHIFT_VALUE_Y

    return SHIFT_VALUE_X, SHIFT_VALUE_Y, SCALE_VALUE_X, SCALE_VALUE_Y 

def shift_and_normalize(batch, shift_value, scale_value):
    return ((batch+shift_value)/scale_value)+CONST_GAMA

def inv_shift_and_normalize(batch, shift_value, scale_value):
    return ((batch-CONST_GAMA)*scale_value)-shift_value