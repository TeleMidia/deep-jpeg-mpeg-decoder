import numpy as np

def add_margin_zeros(data_x, size=8):

    data_x_size = data_x.shape[0]

    dataset_x = []

    zeros_1 = np.zeros((data_x.shape[1], size, 3))
    zeros_2 = np.zeros((8,data_x.shape[2]+size, 3))

    for i_nd in range(0,data_x_size):   
        tensor_x = np.hstack([data_x[i_nd], zeros_1])
        tensor_x = np.vstack([tensor_x, zeros_2])
        dataset_x.append(tensor_x)

    return np.array(dataset_x)

def remove_margin_zeros(data_x, size=8):

    data_x_size = data_x.shape[0]

    height = data_x.shape[1]
    width = data_x.shape[2]
    dataset_x = []

    for i_nd in range(0,data_x_size):
        tensor_x = data_x[i_nd,:(height-8),:,:]
        tensor_x = tensor_x[:,:(width-8),:] 
        dataset_x.append(tensor_x)

    return np.array(dataset_x)           

tensor_x = np.zeros( (1,120,120,3))

tensor_x = add_margin_zeros(tensor_x)
print(tensor_x.shape)
tensor_x = remove_margin_zeros(tensor_x)
print(tensor_x.shape)