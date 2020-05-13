'''
Deep Jpeg Decoder
Copyright (C) 2018 by Antonio J. Grandson Busson <busson@outlook.com>
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

import cv2
import numpy as np
import math
from scipy import fftpack

IMG_DEFAULT_SIZE = 96


zigzag_table = np.array([[0, 1, 5, 6,14,15,27,28],
                         [2, 4, 7,13,16,26,29,42],
                         [3, 8,12,17,25,30,41,43],
                         [9,11,18,24,31,40,44,53],
                         [10,19,23,32,39,45,52,54],
                         [20,22,33,38,46,51,55,60],
                         [21,34,37,47,50,56,59,61],
                         [35,36,48,49,57,58,62,63]])

zigzag_vec_to_block = np.array([[0,0],[0,1],[1,0],[2,0],[1,1],[0,2],[0,3],[1,2],
                               [2,1],[3,0],[4,0],[3,1],[2,2],[1,3],[0,4],[0,5],
                               [1,4],[2,3],[3,2],[4,1],[5,0],[6,0],[5,1],[4,2],
                               [3,3],[2,4],[1,5],[0,6],[0,7],[1,6],[2,5],[3,4],
                               [4,3],[5,2],[6,1],[7,0],[7,1],[6,2],[5,3],[4,4],
                               [3,5],[2,6],[1,7],[2,7],[3,6],[4,5],[5,4],[6,3],
                               [7,2],[7,3],[6,4],[5,5],[4,6],[3,7],[4,7],[5,6],
                               [6,5],[7,4],[7,5],[6,6],[5,7],[6,7],[7,6],[7,7]])


qtable_luma = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                        [12, 12, 14, 19, 26, 58, 60, 55],
                        [14, 13, 16, 24, 40, 57, 69, 56],
                        [14, 17, 22, 29, 51, 87, 80, 62],
                        [18, 22, 37, 56, 68, 109, 103, 77],
                        [24, 35, 55, 64, 81, 104, 113, 92],
                        [49, 64, 78, 87, 103, 121, 120, 101],
                        [72, 92, 95, 98, 112, 100, 103, 99]])

qtable_chroma = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                          [18, 21, 26, 66, 99, 99, 99, 99],
                          [24, 26, 56, 99, 99, 99, 99, 99],
                          [47, 66, 99, 99, 99, 99, 99, 99],
                          [99, 99, 99, 99, 99, 99, 99, 99], 
                          [99, 99, 99, 99, 99, 99, 99, 99],
                          [99, 99, 99, 99, 99, 99, 99, 99],
                          [99, 99, 99, 99, 99, 99, 99, 99]])


def level_shift(block, desloc, data_type=np.int8):
    b_w = block.shape[1]
    b_h = block.shape[0]
    block = block.astype(data_type)
    for x in range(b_w):
        for y in range(b_h):
            block[y][x] = block[y][x] + desloc
    return block

def get_2D_dct(image_block):
    return fftpack.dct(fftpack.dct(image_block.T, norm='ortho').T, norm='ortho')

def get_2D_idct(coefficients_block):
    return fftpack.idct(fftpack.idct(coefficients_block.T, norm='ortho').T, norm='ortho')

def apply_dct(original_block):
    #shifted_block = level_shift(original_block, -128)
    return get_2D_dct(original_block)

def restore_block(dct_block):
    block = get_2D_idct(dct_block)
    block = block.clip(0, 255)
    #block = block.clip(-128, 127)
    #block = level_shift(block, 128, data_type=np.uint8)
    return block

def zigzag_sort(block):
    global zigzag_table
    zigzag_array = np.zeros(64)
   
    for i in range(8):
        for j in range(8):
            zigzag_array[zigzag_table[i][j]] = block[i][j]
        
    return zigzag_array

def zigzag_to_block(zigzag_vector):
    global zigzag_vec_to_block
    block = np.zeros((8,8))
    
    for i in range(64):
        pos = zigzag_vec_to_block[i]
        block[pos[0],pos[1]] = zigzag_vector[i]
        
    return block

def get_dct_image(image):
    width = image.shape[0]
    height = image.shape[1]
    
    dct_image = np.zeros((width,height, 3), np.float32)
    
    for pos_x in range(0, width, 8):
        for pos_y in range(0, height, 8):
            for channel in range(3):
                block = image[pos_y:pos_y+8, pos_x:pos_x+8, channel]
                dct_image[pos_y:pos_y+8, pos_x:pos_x+8, channel] = apply_dct(block)
    
    return dct_image

def restore_image(dct_image):
    width = dct_image.shape[0]
    height = dct_image.shape[1]
    
    image = np.zeros((width,height, 3), np.uint8)
    
    for pos_x in range(0, width, 8):
        for pos_y in range(0, height, 8):
            for channel in range(3):
                block = dct_image[pos_y:pos_y+8, pos_x:pos_x+8, channel]
                image[pos_y:pos_y+8, pos_x:pos_x+8, channel] = restore_block(block)
    
    return image

def clear_coefficients(zigzag_array, begin, end):
    if begin < 0:
        begin = 0
    if end > 64:
        end = 64
    zigzag_array[begin:end] = 0
    return zigzag_array

def clear_image_dct_coefficients(dct_image, coef_begin, coef_end):
    width = dct_image.shape[0]
    height = dct_image.shape[1]
    
    new_dct_image = np.zeros((width,height, 3), np.float32)
    
    for pos_x in range(0, width, 8):
        for pos_y in range(0, height, 8):
            for channel in range(3):
                block = dct_image[pos_y:pos_y+8, pos_x:pos_x+8, channel]
                zigzag_array = zigzag_sort(block)
                zigzag_array = clear_coefficients(zigzag_array, coef_begin, coef_end)
                block = zigzag_to_block(zigzag_array)
                new_dct_image[pos_y:pos_y+8, pos_x:pos_x+8, channel] = block
    
    return new_dct_image

def quantize_dct_block(dtc_block, channel, qtable_luma, qtable_chroma):
    for i in range(0,8):
        for j in range(0,8):
            if channel == 0:
                dtc_block[i][j] = np.round((dtc_block[i][j]/qtable_luma[i][j]), 0)
            else:
                dtc_block[i][j] = np.round((dtc_block[i][j]/qtable_chroma[i][j]), 0)
    
    return dtc_block

def unquantize_dct_block(quantized_block, channel, qtable_luma, qtable_chroma):
    for i in range(0,8):
        for j in range(0,8):
            if channel == 0:
                quantized_block[i][j] = np.round((quantized_block[i][j]*qtable_luma[i][j]), 0)
            else:
                quantized_block[i][j] = np.round((quantized_block[i][j]*qtable_chroma[i][j]), 0)
            
    return quantized_block

def create_zizag_weights(width, height):
    global zigzag_table
    new_image = np.zeros((width,height), np.float32)

    for pos_x in range(0, width, 8):
        for pos_y in range(0, height, 8):
            new_image[pos_y:pos_y+8, pos_x:pos_x+8] = ((zigzag_table-64)*-1)/32


    return new_image

def quantization_dct_image(dct_image, qtable_luma, qtable_chroma, op="quantize"):
    width = dct_image.shape[0]
    height = dct_image.shape[1]
    
    new_dct_image = np.zeros((width,height, 3), np.float32)
    
    for pos_x in range(0, width, 8):
        for pos_y in range(0, height, 8):
            for channel in range(3):
                block = dct_image[pos_y:pos_y+8, pos_x:pos_x+8, channel]
                if op == "quantize":
                    block = quantize_dct_block(block, channel, qtable_luma, qtable_chroma)
                else:
                    block = unquantize_dct_block(block, channel, qtable_luma, qtable_chroma)
                new_dct_image[pos_y:pos_y+8, pos_x:pos_x+8, channel] = block
    
    return new_dct_image  


def scale_qtable(Q):
    global qtable_luma, qtable_chroma

    S = 0
    if Q < 50:
        S = 5000/Q
    else:
        S = 200 - 2*Q

    new_qtable_luma = (qtable_luma*S + 50)/100
    new_qtable_chroma = (qtable_chroma*S + 50)/100

    for i in range(0,8):
        for j in range(0,8):
            if new_qtable_luma[i,j] < 1:
                new_qtable_luma[i,j] = 1
            if new_qtable_chroma[i,j] < 1:
                new_qtable_chroma[i,j] = 1

    return new_qtable_luma.astype(int), new_qtable_chroma.astype(int)


def generate_qtables(quality_factor=50):
    new_qtable_luma, new_qtable_chroma = scale_qtable(quality_factor)
    return new_qtable_luma, new_qtable_chroma

def encode_image(image_data, qtable_luma, qtable_chroma):
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2YUV)

    dct_image = get_dct_image(image_data)
    dct_image = quantization_dct_image(dct_image, qtable_luma, qtable_chroma, op="quantize")
    
    return dct_image

def decode_image(dct_image, qtable_luma, qtable_chroma, in_bgr=True):
    dct_image = quantization_dct_image(dct_image, qtable_luma, qtable_chroma, op="unquantize")
    rest_image = restore_image(dct_image)

    if in_bgr:
        rest_image = cv2.cvtColor(rest_image, cv2.COLOR_YUV2BGR)

    return rest_image