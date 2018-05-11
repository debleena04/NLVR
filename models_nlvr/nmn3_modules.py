from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow import convert_to_tensor as to_T
from sklearn.preprocessing import MinMaxScaler

from models_nlvr.cnn import fc_layer as fc, conv_layer as conv
from models_nlvr.empty_safe_conv import empty_safe_1x1_conv as _1x1_conv
from models_nlvr.empty_safe_conv import empty_safe_conv as _conv

flag = False

def add_spatial_coordinate_map(image_feat_grid):
    image_feat_shape = tf.shape(image_feat_grid)
    N = image_feat_shape[0]
    H = image_feat_shape[1]
    W = image_feat_shape[2]
    x_map = tf.tile(
        tf.reshape(tf.linspace(-1., 1., W), [1, 1, -1, 1]),
        to_T([N, H, 1, 1]))
    y_map = tf.tile(
        tf.reshape(tf.linspace(-1., 1., H), [1, -1, 1, 1]),
        to_T([N, 1, W, 1]))
    # stop gradient on coords_map (needed to fix the tile grad error on TF 1.0.0)
    coords_map = tf.stop_gradient(tf.concat([x_map, y_map], axis=3))
    image_feat_with_coords = tf.concat([image_feat_grid, coords_map], axis=3)
    # set shapes of the new feature maps
    image_feat_static_shape = image_feat_grid.get_shape().as_list()
    image_feat_static_shape[3] += 2
    image_feat_with_coords.set_shape(image_feat_static_shape)
    image_feat_static_shape[3] = 2
    coords_map.set_shape(image_feat_static_shape)
    return image_feat_with_coords, coords_map

class Modules:
    def __init__(self, image_feat_grid, word_vecs, encoder_states, num_choices, map_dim):
        self.image_feat_grid_with_coords, self.coords_map = \
            add_spatial_coordinate_map(image_feat_grid)
        self.word_vecs = word_vecs
        self.encoder_states = encoder_states
        self.num_choices = num_choices
        self.map_dim = map_dim
        # Capture the variable scope for creating all variables
        with tf.variable_scope('module_variables') as module_variable_scope:
            self.module_variable_scope = module_variable_scope
        # Flatten word vecs for efficient slicing
        # word_vecs has shape [T_decoder, N, D]
        word_vecs_shape = tf.shape(word_vecs)
        T_full = word_vecs_shape[0]
        self.N_full = word_vecs_shape[1]
        D_word = word_vecs.get_shape().as_list()[-1]
        self.word_vecs_flat = tf.reshape(
            word_vecs, to_T([T_full*self.N_full, D_word]))

        # create each dummy modules here so that weights won't get initialized again
        att_shape = self.image_feat_grid_with_coords.get_shape().as_list()[:-1] + [1]
        self.att_shape = att_shape
        input_att = tf.placeholder(tf.float32, att_shape)
        time_idx = tf.placeholder(tf.int32, [None])
        batch_idx = tf.placeholder(tf.int32, [None])
        input_vector = tf.placeholder(tf.float32, [None])
        self.FindModule(time_idx, batch_idx,map_dim, reuse=False)
        self.TransformModule(input_att, time_idx, batch_idx, map_dim, reuse=False)
        self.AndModule(input_att, input_att, time_idx, batch_idx, reuse=False)
        self.OrModule(input_att, input_att, time_idx, batch_idx, reuse=False)
        self.NotModule(input_att, time_idx, batch_idx, reuse=False)
        self.DescribeModule(input_att, time_idx, batch_idx, reuse=False)
        self.CountModule(input_att, time_idx, batch_idx, reuse=False)
        self.SamePropertyModule(input_att, input_att, time_idx, batch_idx, reuse=False)
        self.BreakModule(time_idx, batch_idx, reuse=False)
        self.AttReduceModule(input_att, time_idx, batch_idx, reuse=False)
        self.CompareModule(input_vector, time_idx, batch_idx, reuse=False)
        self.CompareReduceModule(input_vector, input_vector, time_idx, batch_idx, reuse=False)
        self.CompareAttModule(input_att, input_att, time_idx, batch_idx, reuse=False)
        self.CombineModule(input_vector, input_vector, input_vector, time_idx, batch_idx, reuse=False)
        self.ExistAttModule(input_vector, input_vector, input_vector, time_idx, batch_idx, reuse=False)
        self.ExistModule(input_vector, time_idx, batch_idx, reuse=False)

    def _slice_image_feat_grid(self, batch_idx):
        # In TF Fold, batch_idx is a [N_batch, 1] tensor
        return tf.gather(self.image_feat_grid_with_coords, batch_idx)

    def _slice_coords_grid(self, batch_idx):
        # In TF Fold, batch_idx is a [N_batch, 1] tensor
        return tf.gather(self.coords_map, batch_idx)

    def _slice_word_vecs(self, time_idx, batch_idx):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors
        # time is highest dim in word_vecs
        joint_index = time_idx*self.N_full + batch_idx
        return tf.gather(self.word_vecs_flat, joint_index)

    def _slice_encoder_states(self, batch_idx):
        # In TF Fold, batch_idx is a [N_batch, 1] tensor
        if self.encoder_states is not None:
            return tf.gather(self.encoder_states, batch_idx)
        else:
            return None
       
    def FindModule(self, time_idx, batch_idx, map_dim=500, scope='FindModule',
        reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        image_feat_grid = self._slice_image_feat_grid(batch_idx)
        text_param = self._slice_word_vecs(time_idx, batch_idx)
        # Mapping: image_feat_grid x text_param -> att_grid
        # Input:
        #   image_feat_grid: [N, H, W, D_im]
        #   text_param: [N, D_txt]
        # Output:
        #   att_grid: [N, H, W, 1]
        #
        # Implementation:
        #   1. Elementwise multiplication between image_feat_grid and text_param
        #   2. L2-normalization
        #   3. Linear classification
        if flag == True:
            image_feat_grid = tf.pad(image_feat_grid, tf.convert_to_tensor([[0,0],[0,0],[1,1],[0,0]]),'CONSTANT')
            image_feat_grid_arr=[]
            image_feat_grid_arr0, image_feat_grid_arr1, image_feat_grid_arr2 = tf.split(image_feat_grid, num_or_size_splits=3, axis=2)
            image_feat_grid_arr.append(image_feat_grid_arr0)
            image_feat_grid_arr.append(image_feat_grid_arr1)
            image_feat_grid_arr.append(image_feat_grid_arr2)
            att_grid_arr = []
            for i in range(3):
                with tf.variable_scope(self.module_variable_scope):
                    with tf.variable_scope(scope, reuse=reuse):
                        image_shape = tf.shape(image_feat_grid[i])
                        N = tf.shape(time_idx)[0]
                        H = image_shape[1]
                        W = image_shape[2]
                        D_im = image_feat_grid.get_shape().as_list()[-1]
                        D_txt = text_param.get_shape().as_list()[-1]

                        # image_feat_mapped has shape [N, H, W, map_dim]
                        image_feat_mapped = _1x1_conv('conv_image', image_feat_grid,
                                              output_dim=map_dim)

                        text_param_mapped = fc('fc_text', text_param, output_dim=map_dim)
                        text_param_mapped = tf.reshape(text_param_mapped, to_T([N, 1, 1, map_dim]))

                        eltwise_mult = tf.nn.l2_normalize(image_feat_mapped * text_param_mapped, 3)
                        att_grid = _1x1_conv('conv_eltwise', eltwise_mult, output_dim=1)

                att_grid.set_shape(self.att_shape)
                att_grid = tf.expand_dims(att_grid,0)
                att_grid = tf.tile(att_grid,[1,1,1,3,1])
                att_grid_arr.append(att_grid)
            return att_grid_arr
            
        if flag == False:
            
            with tf.variable_scope(self.module_variable_scope):
                with tf.variable_scope(scope, reuse=reuse):
                    image_shape = tf.shape(image_feat_grid)
                    N = tf.shape(time_idx)[0]
                    H = image_shape[1]
                    W = image_shape[2]
                    D_im = image_feat_grid.get_shape().as_list()[-1]
                    D_txt = text_param.get_shape().as_list()[-1]

                    # image_feat_mapped has shape [N, H, W, map_dim]
                    image_feat_mapped = _1x1_conv('conv_image', image_feat_grid,
                                              output_dim=map_dim)

                    text_param_mapped = fc('fc_text', text_param, output_dim=map_dim)
                    text_param_mapped = tf.reshape(text_param_mapped, to_T([N, 1, 1, map_dim]))

                    eltwise_mult = tf.nn.l2_normalize(image_feat_mapped * text_param_mapped, 3)
                    att_grid = _1x1_conv('conv_eltwise', eltwise_mult, output_dim=1)

            att_grid.set_shape(self.att_shape)
            att_grid = tf.expand_dims(att_grid,0)
            att_grid = tf.tile(att_grid,[3,1,1,1,1])
            return att_grid

    def TransformModule(self, input_0, time_idx, batch_idx, kernel_size=5,
        map_dim=500, scope='TransformModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        image_feat_grid = self._slice_image_feat_grid(batch_idx)
        text_param = self._slice_word_vecs(time_idx, batch_idx)
        # Mapping: att_grid x text_param -> att_grid
        # Input:
        #   input_0: [N, H, W, 1]
        #   text_param: [N, D_txt]
        # Output:
        #   att_grid: [N, H, W, 1]
        #
        # Implementation (Same as FindSamePropertyModule):
        #   1. Extract visual features using the input attention map, and
        #      linear transform to map_dim
        #   2. linear transform language features to map_dim
        #   3. Convolve image features to map_dim
        #   4. Element-wise multiplication of the three, l2_normalize, linear transform.
        with tf.variable_scope(self.module_variable_scope):
            with tf.variable_scope(scope, reuse=reuse):
                image_shape = tf.shape(image_feat_grid)
                N = tf.shape(time_idx)[0]
                H = image_shape[1]
                W = image_shape[2]
                D_im = image_feat_grid.get_shape().as_list()[-1]
                D_txt = text_param.get_shape().as_list()[-1]

                # image_feat_mapped has shape [N, H, W, map_dim]
                image_feat_mapped = _1x1_conv('conv_image', image_feat_grid,
                                              output_dim=map_dim)

                text_param_mapped = fc('fc_text', text_param, output_dim=map_dim)
                text_param_mapped = tf.reshape(text_param_mapped, to_T([N, 1, 1, map_dim]))

                att_softmax = tf.reshape(
                    tf.nn.softmax(tf.reshape(input_0, to_T([N, H*W]))),
                    to_T([N, H, W, 1]))
                # att_feat has shape [N, D_vis]
                att_feat = tf.reduce_sum(image_feat_grid * att_softmax, axis=[1, 2])
                att_feat_mapped = tf.reshape(
                    fc('fc_att', att_feat, output_dim=map_dim), to_T([N, 1, 1, map_dim]))

                eltwise_mult = tf.nn.l2_normalize(
                    image_feat_mapped * text_param_mapped * att_feat_mapped, 3)
                att_grid = _1x1_conv('conv_eltwise', eltwise_mult, output_dim=1)

        att_grid.set_shape(self.att_shape)
        return att_grid

    

    def DescribeModule(self, input_0, time_idx, batch_idx, scope='DescribeModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        att_grid = input_0
        # Mapping: att_grid -> answer probs
        # Input:
        #   att_grid: [N, H, W, 1]
        # Output:
        #   answer_scores: [N, self.num_choices]
        #
        # Implementation:
        #   1. Max-pool over att_grid
        #   2. a linear mapping layer (without ReLU)
        with tf.variable_scope(scope, reuse=reuse):
            att_shape = tf.shape(att_grid)
            N = att_shape[0]
            H = att_shape[1]
            W = att_shape[2]

            att_min = tf.reduce_min(att_grid, axis=[1, 2])
            att_avg = tf.reduce_mean(att_grid, axis=[1, 2])
            att_max = tf.reduce_max(att_grid, axis=[1, 2])
            # att_reduced has shape [N, 3]
            att_reduced = tf.concat([att_min, att_avg, att_max], axis=1)
            scores = fc('fc_scores', att_reduced, output_dim=self.num_choices)
            scores = tf.nn.softmax(scores)
        return scores
    
    def AndModule(self, input_0, input_1, time_idx, batch_idx,
        scope='AndModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        # Mapping: att_grid x att_grid -> att_grid
        # Input:
        #   input_0: [N, H, W, 1]
        #   input_1: [N, H, W, 1]
        # Output:
        #   att_grid: [N, H, W, 1]
        #
        # Implementation:
        #   Take the elementwise-min
        with tf.variable_scope(self.module_variable_scope):
            with tf.variable_scope(scope, reuse=reuse):
                att_grid = tf.minimum(input_0, input_1)

        att_grid.set_shape(self.att_shape)
        return att_grid

    def OrModule(self, input_0, input_1, time_idx, batch_idx,
        scope='OrModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        # Mapping: att_grid x att_grid -> att_grid
        # Input:
        #   input_0: [N, H, W, 1]
        #   input_1: [N, H, W, 1]
        # Output:
        #   att_grid: [N, H, W, 1]
        #
        # Implementation:
        #   Take the elementwise-max
        with tf.variable_scope(self.module_variable_scope):
            with tf.variable_scope(scope, reuse=reuse):
                att_grid = tf.maximum(input_0, input_1)

        att_grid.set_shape(self.att_shape)
        return att_grid
    
    def NotModule(self, input_0, time_idx, batch_idx,
        scope='NotModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        # Mapping: att_grid  -> att_grid
        # Input:
        #   input_0: [N, H, W, 1]
        
        # Output:
        #   att_grid: [N, H, W, 1]
        #
        # Implementation:
        #   Take the 1 - attention value
        min_d = tf.reduce_min(input_0)
        max_d = tf.reduce_max(input_0)
        att_normalised = (input_0 - min_d) / (max_d - min_d)
        with tf.variable_scope(self.module_variable_scope):
            with tf.variable_scope(scope, reuse=reuse):
                att_grid = 1 - att_normalised

        att_grid_reshape = att_normalised * (max_d - min_d) + min_d
        return att_grid_reshape

       
    
    def CountModule(self, input_0, time_idx, batch_idx,map_dim=500,
        scope='CountModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        # Mapping: att_grid -> answer probs
        # Input:
        #   input_0: [N, H, W, 1]
        # Output:
        #   answer_scores: [N, self.num_choices]
        #
        # Implementation:
        #   1. linear transform of the attention map (also including max and min)
        with tf.variable_scope(self.module_variable_scope):
            with tf.variable_scope(scope, reuse=reuse):
                H, W = self.att_shape[1:3]
                att_all = tf.reshape(input_0, to_T([-1, H*W]))
                att_min = tf.reduce_min(input_0, axis=[1, 2])
                att_max = tf.reduce_max(input_0, axis=[1, 2])
                # att_reduced has shape [N, 3]
                att_concat = tf.concat([att_all, att_min, att_max], axis=1)
                scores = fc('fc_scores', att_concat, output_dim=map_dim)
        return scores
    
    def SamePropertyModule(self, input_0, input_1, time_idx, batch_idx,
        map_dim=500, scope='SamePropertyModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        image_feat_grid = self._slice_image_feat_grid(batch_idx)
        text_param = self._slice_word_vecs(time_idx, batch_idx)
        # Mapping: att_grid x att_grid -> answer probs
        # Input:
        #   input_0: [N, H, W, 1]
        #   input_1: [N, H, W, 1]
        # Output:
        #   answer_scores: [N, self.num_choices]
        #
        # Implementation:
        #   1. Extract visual features using the input attention map, and
        #      linear transform to map_dim
        #   2. linear transform language features to map_dim
        #   3. Convolve image features to map_dim
        #   4. Element-wise multiplication of the three, l2_normalize, linear transform.
        with tf.variable_scope(self.module_variable_scope):
            with tf.variable_scope(scope, reuse=reuse):
                image_shape = tf.shape(image_feat_grid)
                N = tf.shape(time_idx)[0]
                H = image_shape[1]
                W = image_shape[2]
                D_im = image_feat_grid.get_shape().as_list()[-1]
                D_txt = text_param.get_shape().as_list()[-1]

                text_param_mapped = fc('fc_text', text_param, output_dim=map_dim)

                att_softmax_0 = tf.reshape(
                    tf.nn.softmax(tf.reshape(input_0, to_T([N, H*W]))),
                    to_T([N, H, W, 1]))
                att_softmax_1 = tf.reshape(
                    tf.nn.softmax(tf.reshape(input_1, to_T([N, H*W]))),
                    to_T([N, H, W, 1]))
                # att_feat_0, att_feat_1 has shape [N, D_vis]
                att_feat_0 = tf.reduce_sum(image_feat_grid * att_softmax_0, axis=[1, 2])
                att_feat_1 = tf.reduce_sum(image_feat_grid * att_softmax_1, axis=[1, 2])
                att_feat_mapped_0 = tf.reshape(
                    fc('fc_att_0', att_feat_0, output_dim=map_dim),
                    to_T([N, map_dim]))
                att_feat_mapped_1 = tf.reshape(
                    fc('fc_att_1', att_feat_1, output_dim=map_dim),
                    to_T([N, map_dim]))

                eltwise_mult = tf.nn.l2_normalize(
                    att_feat_mapped_0 * text_param_mapped * att_feat_mapped_1, 1)
                scores = fc('fc_eltwise', eltwise_mult, output_dim=map_dim)

        return scores
   
    def BreakModule(self, time_idx, batch_idx, map_dim=500, scope='BreakModule',
        reuse=True):
        # In TF Fold, batch_idx is [N_batch, 1] tensors
        
        
        # Mapping: image_feat_grid -> image_feat_array
        # Input:
        #   image_feat_grid: [N, H, W, D_im]
        
        # Output:
        #   image_feat_array: [N,H,100,3]
        #
        # Implementation:
        #   1. Break the feature grid into 3 feature array anfd concat them into one single array
        flag = True 
            
        
    
    def AttReduceModule(self, input_0, time_idx, batch_idx, map_dim=500,
        scope='AttReduceModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        # Mapping: att_grid -> answer probs
        # Input:
        #   att_grid: [N, H, W, 1]
        # Output:
        #   answer_scores: [N, self.num_choices]
        #
        # Implementation:
        #   1. Max-pool over att_grid
        #   2. a linear mapping layer (without ReLU)
        with tf.variable_scope(self.module_variable_scope):
            with tf.variable_scope(scope, reuse=reuse):
                att_min = tf.reduce_min(input_0, axis=[1, 2])
                att_avg = tf.reduce_mean(input_0, axis=[1, 2])
                att_max = tf.reduce_max(input_0, axis=[1, 2])
                # att_reduced has shape [N, 3]
                att_reduced = tf.concat([att_min, att_avg, att_max], axis=1)
                #scores = fc('fc_scores', att_reduced, output_dim=map_dim)
        return att_reduced
    
    def CompareModule(self, input_0, time_idx, batch_idx, map_dim=500, scope='CompareModule',
        reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors
        # input0 is the vector output from Count module
        image_feat_grid = self._slice_image_feat_grid(batch_idx)
        text_param = self._slice_word_vecs(time_idx, batch_idx)
        # Mapping: input0 x text_param -> scores
        # Input:
        #   input0: [N, 1]
        #   text_param: [N, D_txt]
        # Output:
        #   vector : [N,1]
        #
        # Implementation:
        #   1. Elementwise multiplication between image_feat_grid and text_param
        #   2. L2-normalization
        #   3. Linear classification
        text_param_mapped = fc('fc_text', text_param, output_dim=map_dim)
        vector_mapped = fc('input_vector_0',input_0, output_dim = map_dim)
        scores = fc('fc_eltwise', text_param_mapped + vector_mapped, output_dim=map_dim)
        #scores = text_param_mapped + vector_mapped
        return scores
   
    def CompareReduceModule(self, input_0, input_1, time_idx, batch_idx, map_dim=500, scope='CompareReduceModule',
        reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors
        # input0 is the vector output from Count module
        # input1 is the vector output from count module
        # Mapping: input0 x text_param -> scores
        # Input:
        #   input0 : [N, 1]
        #   input1 : [N, 1]
        # Output:
        #   vector : [N,1]
        #
        # Implementation:
        #   1. Elementwise multiplication between image_feat_grid and text_param
        #   2. L2-normalization
        #   3. Linear classification
        #vector0_mapped = fc('input_vector0',input0, output_dim = map_dim)
        #vector1_mapped = fc('input_vector1',input1, output_dim = map_dim)
        scores = fc('fc_eltwise', input_0 + input_1, output_dim=map_dim)
        return scores
    
    def CompareAttModule(self, input_0, input_1, time_idx, batch_idx, map_dim=500, scope='CompareAttModule',
        reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors
        # input0 is the vector output from Count module
        # input1 is the vector output from count module
        # Mapping: input0 x text_param -> scores
        # Input:
        #   input0 : [N, 1]
        #   input1 : [N, 1]
        # Output:
        #   vector : [N,1]
        #
        # Implementation:
        #   1. Elementwise multiplication between image_feat_grid and text_param
        #   2. L2-normalization
        #   3. Linear classification
        N = tf.shape(time_idx)[0]
        att_feat0_mapped = tf.reshape(
                    fc('fc_att', input_0, output_dim=map_dim),
                    to_T([N, map_dim]))
        att_feat1_mapped = tf.reshape(
                    fc('fc_att', input_1, output_dim=map_dim),
                    to_T([N, map_dim]))
        scores = fc('fc_eltwise', att_feat0_mapped + att_feat1_mapped, output_dim=map_dim)
        return scores
    
    def CombineModule(self, input_0, input_1, input_2, time_idx, batch_idx, map_dim=500, scope='CombineModule',
        reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors
        # input0 is the vector output to be combined
        # input1 is the vector output to be combined
        # input2 is the vector output to be combined
        # Mapping: input0 x text_param -> scores
        # Input:
        #   input0 : [N, 1]
        #   input1 : [N, 1]
        #   input2 : [N, 1]
        # Output:
        #   vector : [N,1]
        #
        # Implementation:
        #   1. Elementwise multiplication between image_feat_grid and text_param
        #   2. L2-normalization
        #   3. Linear classification
        N = tf.shape(time_idx)[0]
        text_param = self._slice_word_vecs(time_idx, batch_idx)
        text_param_mapped = fc('fc_text', text_param, output_dim=3)
        text_param_softmax = tf.nn.softmax(text_param_mapped)
        with tf.variable_scope('token_prediction'):
                w_y = tf.get_variable('weights', [len(input_0),1],
                    initializer=tf.contrib.layers.xavier_initializer())
                b_y = tf.get_variable('biases', 1,
                    initializer=tf.constant_initializer(0.))
                input_0_mapped = tf.sigmoid(tf.nn.xw_plus_b(input_0, w_y, b_y))
                input_1_mapped = tf.sigmoid(tf.nn.xw_plus_b(input_1, w_y, b_y))
                input_2_mapped = tf.sigmoid(tf.nn.xw_plus_b(input_2, w_y, b_y))
                scores_matrix = [tf.miminum(input_0_mapped, input_1_mapped, input_2_mapped),
                  tf.maximum(input_0_mapped, input_1_mapped, input_2_mapped),
                  tf.maximum(tf.minimum(input_0_mapped,input_1_mapped), tf.minimum(input_0_mapped,input_2_mapped), tf.minimum(input_2_mapped,input_1_mapped))]
                score = tf.matmul(text_param_softmax,tf.matrix_transpose(scores_matrix))
                scores = tf.convert_to_tensor([score,1-score])
        return scores
    
    def ExistAttModule(self, input_0, input_1, input_2, time_idx, batch_idx, map_dim=500, scope='ExistAttModule',
        reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors
        # input0 is the attention output to be combined
        # input1 is the attention output to be combined
        # input2 is the attention output to be combined
        # Mapping: input0 x text_param -> scores
        # Input:
        #   input0 : [N, 1]
        #   input1 : [N, 1]
        #   input2 : [N, 1]
        # Output:
        #   vector : [N,1]
        #
        # Implementation:
        #   1. Elementwise multiplication between image_feat_grid and text_param
        #   2. L2-normalization
        #   3. Linear classification
        N = tf.shape(time_idx)[0]
        text_param = self._slice_word_vecs(time_idx, batch_idx)
        text_param_mapped = fc('fc_text', text_param, output_dim=3)
        text_param_softmax = tf.nn.softmax(text_param_mapped)
        with tf.variable_scope('token_prediction'):
                w_y = tf.get_variable('weights', [len(input_0),1],
                    initializer=tf.contrib.layers.xavier_initializer())
                b_y = tf.get_variable('biases', 1,
                    initializer=tf.constant_initializer(0.))
                input_0_mapped = tf.sigmoid(tf.nn.xw_plus_b(input_0, w_y, b_y))
                input_1_mapped = tf.sigmoid(tf.nn.xw_plus_b(input_1, w_y, b_y))
                input_2_mapped = tf.sigmoid(tf.nn.xw_plus_b(input_2, w_y, b_y))
                scores_matrix = [tf.miminum(input_0_mapped, input_1_mapped, input_2_mapped),
                  tf.maximum(input_0_mapped, input_1_mapped, input_2_mapped),
                  tf.maximum(tf.minimum(input_0_mapped,input_1_mapped), tf.minimum(input_0_mapped,input_2_mapped), tf.minimum(input_2_mapped,input_1_mapped))]
                score = tf.matmul(text_param_softmax,tf.matrix_transpose(scores_matrix))
                scores = tf.convert_to_tensor([score,1-score])
                scores = fc('fc_scores', scores, output_dim=self.num_choices)
                scores = tf.nn.softmax(scores)
        return scores
    
    def ExistModule(self, input_0, time_idx, batch_idx, scope='ExistModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors

        
        # Mapping: att_grid -> answer probs
        # Input:
        #   att_grid: [N, H, W, 1]
        # Output:
        #   answer_scores: [N, self.num_choices]
        #
        # Implementation:
        #   1. Max-pool over att_grid
        #   2. a linear mapping layer (without ReLU)
        with tf.variable_scope(scope, reuse=reuse):
            scores = fc('fc_scores', input_0, output_dim=self.num_choices)
            scores = tf.nn.softmax(scores)
        return scores
    
    





