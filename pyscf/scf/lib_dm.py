#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 19:08:14 2025

@author: lioneltruflandier
"""

import tensorflow as tf
import numpy as np

def mm(A,B,method='np',dtype='float32'):
    if ( method == 'np'  ): # numpy matmul
        C = A @ B
    if ( method == 'es'  ): # einsum matmul
        C = np.einsum('ik,kj->ij', A, B, optimize=True)
    if ( method == 'tf' ):  # tensor flow/GPU/CPU matmul

        if not ( tf.is_tensor(A) and tf.is_tensor(B) ):
            print("mm/tf: A and B are not tensors!")

        C = tf.matmul(A,B)

    return C

if __name__ == '__main__':

    from time import perf_counter
    
    A = np.random.rand(10000,10000)
    B = np.random.rand(10000,10000)
    #A = np.float32(A)
    #B = np.float32(B)

    # timing of standard matrix multiply with numpy
    t0_np = perf_counter()
    C_np  = mm(A,B,method='np')
    t1_np = perf_counter()

    # timing of the conversion np -> tf
    t0_tf_convert = perf_counter()
    A_tf = tf.convert_to_tensor(A,dtype='float32')
    B_tf = tf.convert_to_tensor(B,dtype='float32')
    t1_tf_convert = perf_counter()

    # timing of tensor flow matrix multiply
    t0_tf = perf_counter()
    C_tf  = mm(A_tf,B_tf,method='tf')
    t1_tf = perf_counter()

    # timing of the conversion tf -> np
    t0_np_convert = perf_counter()
    C_tf.numpy()
    t1_np_convert = perf_counter()

    #test1 = np.linalg.norm(C_np - C_tf, ord='fro')/np.linalg.norm(C_np)
    #print(test1)
    print('tf time = %.3f + %.3f/%.3f (conv. back/forth) vs. np = %.3f'%((t1_tf-t0_tf),
         (t1_tf_convert-t0_tf_convert),(t1_np_convert-t0_np_convert),(t1_np-t0_np)))