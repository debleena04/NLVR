from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow import convert_to_tensor as to_T
from sklearn.preprocessing import MinMaxScaler

from models_nlvr_new.cnn import fc_layer as fc, conv_layer as conv
from models_nlvr_new.empty_safe_conv import empty_safe_1x1_conv as _1x1_conv
from models_nlvr_new.empty_safe_conv import empty_safe_conv as _conv

#flag = False

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
    def __init__(self, image_feat_grid, word_vecs, encoder_states, num_choices, map_dim, batch_size):
        self.image_feat_grid_with_coords, self.coords_map = \
            add_spatial_coordinate_map(image_feat_grid)
        self.word_vecs = word_vecs
        self.encoder_states = encoder_states
        self.num_choices = num_choices
        self.map_dim = map_dim
        self.flag = True
        self.batch_size = batch_size

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
        att_shape[2]+=2
        flatten_att_shape = [att_shape[0]]+[3]+att_shape[1:]
        self.flatten_att_shape = flatten_att_shape
        self.att_shape = att_shape
        vector_shape = [att_shape[0]]+[3,map_dim]
        self.vector_shape = vector_shape
        input_att = tf.placeholder(tf.float32, flatten_att_shape)
        time_idx = tf.placeholder(tf.int32, [None])
        batch_idx = tf.placeholder(tf.int32, [None]) 
        input_vector = tf.placeholder(tf.float32, self.vector_shape)
        self.FindModule(time_idx, batch_idx,map_dim, reuse=False)
        self.TransformModule(input_att, time_idx, batch_idx, map_dim, reuse=False)
        self.AndModule(input_att, input_att, time_idx, batch_idx, reuse=False)
        self.OrModule(input_att, input_att, time_idx, batch_idx, reuse=False)
        self.NotModule(input_att, time_idx, batch_idx, reuse=False)
        self.DescribeModule(input_att, time_idx, batch_idx, reuse=False)
        self.CountModule(input_att, time_idx, batch_idx,map_dim, reuse=False)
        self.SamePropertyModule(input_att, input_att, time_idx, batch_idx, map_dim, reuse=False)
        self.BreakModule(time_idx, batch_idx, map_dim, reuse=False)
        self.AttReduceModule(input_att, time_idx, batch_idx, map_dim, reuse=False)
        self.CompareModule(input_vector, time_idx, batch_idx, map_dim, reuse=False)
        self.CompareReduceModule(input_vector, input_vector, time_idx, batch_idx, map_dim, reuse=False)
        self.CompareAttModule(input_att, input_att, time_idx, batch_idx, map_dim, reuse=False)
        self.CombineModule(input_vector, time_idx, batch_idx, map_dim, reuse=False)
        self.ExistAttModule(input_vector, time_idx, batch_idx, map_dim, reuse=False)
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
    

    def BreakModule(self, time_idx, batch_idx, map_dim=500, scope='BreakModule',
        reuse=True):
        print("Flag in break before setting",self.flag)
        self.flag = True
        print("Flag in break after setting",self.flag)
   
    def FindModule(self, time_idx, batch_idx, map_dim=500, scope='FindModule',
        reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors
        print("Flag value in find",self.flag)
        image_feat_grid = self._slice_image_feat_grid(batch_idx)
        text_param = self._slice_word_vecs(time_idx, batch_idx)
        D_im = image_feat_grid.get_shape().as_list()[-1]
        D_txt = text_param.get_shape().as_list()[-1]
        with tf.variable_scope(self.module_variable_scope):
           with tf.variable_scope(scope, reuse=reuse):
               with tf.variable_scope('conv_image'):
                  W_c = tf.get_variable('weights', [D_im, map_dim],
                     initializer=tf.contrib.layers.xavier_initializer())
                  b_c = tf.get_variable('biases', map_dim,
                     initializer=tf.constant_initializer(0.))
               with tf.variable_scope('fc_text'):
                  W_t = tf.get_variable('weights', [D_txt, map_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
                  b_t = tf.get_variable('biases', map_dim,
                     initializer=tf.constant_initializer(0.))
               with tf.variable_scope('conv_eltwise'):
                  W_e = tf.get_variable('weights', [map_dim,1],
                     initializer=tf.contrib.layers.xavier_initializer())
                  b_e = tf.get_variable('biases', 1,
                     initializer=tf.constant_initializer(0.))

        #print("image_feat_grid shape in find:", image_feat_grid.shape)
        #print("text_param shape in find:", text_param.shape)
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
        if self.flag == True:
            image_feat_grid = tf.pad(image_feat_grid, tf.convert_to_tensor([[0,0],[0,0],[1,1],[0,0]]),'CONSTANT')
            #print("image_feat_grid shape in find:", image_feat_grid.shape)
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
                        image_feat_mapped = _1x1_conv('conv_image', image_feat_grid_arr[i],
                                              output_dim=map_dim, reuse = True)
                        image_feat_mapped = tf.reshape(image_feat_mapped, to_T([N, 1, 1, map_dim]))

                        text_param_mapped = fc('fc_text', text_param, output_dim=map_dim, reuse = True)
                        text_param_mapped = tf.reshape(text_param_mapped, to_T([N, 1, 1, map_dim]))

                        eltwise_mult = tf.nn.l2_normalize(image_feat_mapped * text_param_mapped, 3)
                        att_grid = _1x1_conv('conv_eltwise', eltwise_mult, output_dim=1, reuse = True)
                        #print("att grid in find:", att_grid.shape)
                        att_grid= tf.reshape(att_grid, to_T([N,10,14,1]))
                        #att_grid = tf.expand_dims(att_grid,1)
                        #print("att grid shape",att_grid.shape)
                        att_grid = tf.tile(att_grid,[1,1,3,1])
                        att_grid_arr.append(att_grid)
                        #print("att grid shape after tiling",att_grid.shape)
                        #print(att_grid_arr.shape)
            arr = tf.convert_to_tensor(att_grid_arr)
            arr =tf.transpose(arr,[1,0,2,3,4])
            return arr
            
        if self.flag == False:
            #image_feat_grid = tf.expand_dims(image_feat_grid,axis= 0)
            image_feat_grid = tf.pad(image_feat_grid, tf.convert_to_tensor([[0,0],[0,0],[1,1],[0,0]]),'CONSTANT')
            with tf.variable_scope(self.module_variable_scope):
                with tf.variable_scope(scope, reuse=reuse):
                    image_shape = tf.shape(image_feat_grid)
                    N = tf.shape(time_idx)[0]
                    H = image_shape[1]
                    W = image_shape[2]
                    D_im = image_feat_grid.get_shape().as_list()[-1]
                    D_txt = text_param.get_shape().as_list()[-1]
                    
                    #print("image feat shape in find false ",image_feat_grid.shape)
                    #image_feat_grid = tf.expand_dims(image_feat_grid,axis= 0)
                    # image_feat_mapped has shape [N, H, W, map_dim]
                    image_feat_mapped = _1x1_conv('conv_image', image_feat_grid,
                                              output_dim=map_dim, reuse = True)
                    #print("image_feat mapped ",image_feat_mapped.shape)
                    image_feat_mapped = tf.reshape(image_feat_mapped, to_T([N, H, W, map_dim]))
                    #print("image_feat mapped ",image_feat_mapped.shape)
                    text_param_mapped = fc('fc_text', text_param, output_dim=map_dim, reuse = True)
                    text_param_mapped = tf.reshape(text_param_mapped, to_T([N, 1, 1, map_dim]))
                    #print("text_feat mapped ",text_param_mapped.shape)
                    eltwise_mult = tf.nn.l2_normalize(image_feat_mapped * text_param_mapped, 3)
                    #print("eltwise_mult ",eltwise_mult.shape)
                    att_grid = _1x1_conv('conv_eltwise', eltwise_mult, output_dim=1, reuse = True)

                    att_grid.set_shape(self.att_shape)
                    att_grid = tf.expand_dims(att_grid,1)
                    att_grid = tf.tile(att_grid,[1,3,1,1,1])
                    #print("att grid in find false ",att_grid.shape)
            return att_grid

    def TransformModule(self, input_0, time_idx, batch_idx, kernel_size=5,
        map_dim=500, scope='TransformModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors
        image_feat_grid = self._slice_image_feat_grid(batch_idx)
        text_param = self._slice_word_vecs(time_idx, batch_idx)
        D_im = image_feat_grid.get_shape().as_list()[-1]
        D_txt = text_param.get_shape().as_list()[-1]
        with tf.variable_scope(self.module_variable_scope):
           with tf.variable_scope(scope, reuse=reuse):
               with tf.variable_scope('conv_image'):
                  W_c = tf.get_variable('weights', [D_im, map_dim],
                     initializer=tf.contrib.layers.xavier_initializer())
                  b_c = tf.get_variable('biases', map_dim,
                     initializer=tf.constant_initializer(0.))
               with tf.variable_scope('fc_text'):
                  W_t = tf.get_variable('weights', [D_txt, map_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
                  b_t = tf.get_variable('biases', map_dim,
                     initializer=tf.constant_initializer(0.))
               with tf.variable_scope('fc_att'):
                  W_t = tf.get_variable('weights', [D_im, map_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
                  b_t = tf.get_variable('biases', map_dim,
                     initializer=tf.constant_initializer(0.))
               with tf.variable_scope('conv_eltwise'):
                  W_e = tf.get_variable('weights', [map_dim,1],
                     initializer=tf.contrib.layers.xavier_initializer())
                  b_e = tf.get_variable('biases', 1,
                     initializer=tf.constant_initializer(0.))
        
        if self.flag == True:
            #image_feat_grid = tf.pad(image_feat_grid, tf.convert_to_tensor([[0,0],[0,0],[1,1],[0,0]]),'CONSTANT')
            input_arr=[]
            input_arr0, input_arr1, input_arr2 = tf.split(input_0, num_or_size_splits=3, axis=1)
            input_arr.append(input_arr0[:,:,:,:14,:])
            input_arr.append(input_arr1[:,:,:,14:28,:])
            input_arr.append(input_arr2[:,:,:,28:,:])
            image_feat_grid = tf.pad(image_feat_grid, tf.convert_to_tensor([[0,0],[0,0],[1,1],[0,0]]),'CONSTANT')
            #print("image_feat_grid shape in find:", image_feat_grid.shape)
            image_feat_grid_arr=[]
            image_feat_grid_arr0, image_feat_grid_arr1, image_feat_grid_arr2 = tf.split(image_feat_grid, num_or_size_splits=3, axis=2)
            image_feat_grid_arr.append(image_feat_grid_arr0)
            image_feat_grid_arr.append(image_feat_grid_arr1)
            image_feat_grid_arr.append(image_feat_grid_arr2)
            att_grid_arr = []
            for i in range(3):
                 with tf.variable_scope(self.module_variable_scope):
                    with tf.variable_scope(scope, reuse=reuse):
                        image_shape = tf.shape(input_arr[i])
                        N = tf.shape(time_idx)[0]
                        H = image_shape[2]
                        W = image_shape[3]
                        D_im = image_feat_grid.get_shape().as_list()[-1]
                        D_txt = text_param.get_shape().as_list()[-1]

                       # image_feat_mapped has shape [N, H, W, map_dim]
                        image_feat_mapped = _1x1_conv('conv_image', image_feat_grid_arr[i],
                                              output_dim=map_dim, reuse =True)

                        text_param_mapped = fc('fc_text', text_param, output_dim=map_dim, reuse =True)
                        text_param_mapped = tf.reshape(text_param_mapped, to_T([N, 1, 1, map_dim]))

                        att_softmax = tf.reshape(
                            tf.nn.softmax(tf.reshape(input_arr[i], to_T([ N, H*W]))),
                            to_T([ N, H, W, 1]))
                        # att_feat has shape [N, D_vis]
                        att_feat = tf.reduce_sum(image_feat_grid_arr[i] * att_softmax, axis=[1, 2])
                        att_feat_mapped = tf.reshape(
                            fc('fc_att', att_feat, output_dim=map_dim, reuse =True), to_T([N, 1, 1, map_dim]))
 
                        eltwise_mult = tf.nn.l2_normalize(
                            image_feat_mapped * text_param_mapped * att_feat_mapped, 3)
                        att_grid = _1x1_conv('conv_eltwise', eltwise_mult, output_dim=1, reuse =True)
                       
                        att_grid= tf.reshape(att_grid, to_T([N,10,14,1]))
                        #att_grid = tf.expand_dims(att_grid,1)
                        #print("att grid shape",att_grid.shape)
                        att_grid = tf.tile(att_grid,[1,1,3,1])
                        att_grid_arr.append(att_grid)
                        #print("att grid shape after tiling",att_grid.shape)
                        #print(att_grid_arr.shape)
            arr = tf.convert_to_tensor(att_grid_arr)
            arr =tf.transpose(arr,[1,0,2,3,4])
            return arr       
        
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
        if self.flag == False:
            
            input_arr0, input_arr1, input_arr2 = tf.split(input_0, num_or_size_splits=3, axis=1)
            image_feat_grid = tf.pad(image_feat_grid, tf.convert_to_tensor([[0,0],[0,0],[1,1],[0,0]]),'CONSTANT')
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
                                              output_dim=map_dim, reuse =True)

                    text_param_mapped = fc('fc_text', text_param, output_dim=map_dim, reuse =True)
                    text_param_mapped = tf.reshape(text_param_mapped, to_T([N, 1, 1, map_dim]))

                    att_softmax = tf.reshape(
                        tf.nn.softmax(tf.reshape(input_arr0, to_T([N, H*W]))),
                        to_T([ N, H, W, 1]))
                    # att_feat has shape [N, D_vis]
                    att_feat = tf.reduce_sum(image_feat_grid * att_softmax, axis=[1, 2])
                    att_feat_mapped = tf.reshape(
                        fc('fc_att', att_feat, output_dim=map_dim, reuse =True), to_T([ N, 1, 1, map_dim]))

                    eltwise_mult = tf.nn.l2_normalize(
                        image_feat_mapped * text_param_mapped * att_feat_mapped, 3)
                    att_grid = _1x1_conv('conv_eltwise', eltwise_mult, output_dim=1, reuse =True)

                    att_grid.set_shape(self.att_shape)
                    att_grid = tf.expand_dims(att_grid,1)
                    att_grid = tf.tile(att_grid,[1,3,1,1,1])
            return att_grid

    

    def DescribeModule(self, input_0, time_idx, batch_idx, scope='DescribeModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors
        #image_feat_grid = self._slice_image_feat_grid(batch_idx)
        image_feat_grid_arr0, image_feat_grid_arr1, image_feat_grid_arr2 = tf.split(input_0, num_or_size_splits=3, axis=1)
        att_grid = image_feat_grid_arr0
        #print("att grid shape in describe:", att_grid.shape)
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
              att_shape = tf.shape(att_grid)
              N = att_shape[0]
              H = att_shape[2]
              W = att_shape[3]

              att_min = tf.reduce_min(att_grid, axis=[2,3])
              att_avg = tf.reduce_mean(att_grid, axis=[2, 3])
              att_max = tf.reduce_max(att_grid, axis=[2, 3])
              # att_reduced has shape [N, 3]
              #print("shape" , att_min.shape, att_avg.shape,att_max.shape)
              att_reduced = tf.concat([att_min, att_avg, att_max], axis=1)
              #print("att reduced shape in describe ",att_reduced.shape)
              att_reduced = tf.reshape(att_reduced,[-1,3])
              scores = fc('fc_scores', att_reduced, output_dim=self.num_choices)
              scores = tf.nn.softmax(scores)
              self.flag = False
              
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

        #att_grid.set_shape(self.att_shape)
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

        #att_grid.set_shape(self.att_shape)
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
        #print("input shape in count:",input_0.shape)
        with tf.variable_scope(self.module_variable_scope):
                    with tf.variable_scope(scope, reuse=reuse):
                        H, W = self.att_shape[1:3]
                        with tf.variable_scope('fc_text'):
                            W_t = tf.get_variable('weights', [H*W+2, map_dim],
                             initializer=tf.contrib.layers.xavier_initializer())
                            b_t = tf.get_variable('biases', map_dim,
                              initializer=tf.constant_initializer(0.))
        if self.flag == True:
            image_feat_grid_arr=[]
            image_feat_grid_arr0, image_feat_grid_arr1, image_feat_grid_arr2 = tf.split(input_0, num_or_size_splits=3, axis=1)
            image_feat_grid_arr.append(image_feat_grid_arr0[:,:,:,:14,:])
            image_feat_grid_arr.append(image_feat_grid_arr1[:,:,:,14:28,:])
            image_feat_grid_arr.append(image_feat_grid_arr2[:,:,:,28:,:])
            #print("in count:",image_feat_grid_arr[0].shape)
            scores_arr = []
            for i in range(3):
                with tf.variable_scope(self.module_variable_scope):
                    with tf.variable_scope(scope, reuse=reuse):
                        H, W = self.att_shape[1:3]
                        att_all = tf.reshape(image_feat_grid_arr[i], to_T([-1, 1, H*W]))
                        att_min = tf.reduce_min(image_feat_grid_arr[i], axis=[2, 3])
                        att_max = tf.reduce_max(image_feat_grid_arr[i], axis=[2, 3])
                        # att_reduced has shape [N, 3]
                        att_concat = tf.concat([att_all, att_min, att_max], axis=2)
                        att_concat = tf.reshape(att_concat,[-1,H*W+2])
                        scores = fc('fc_scores', att_concat, output_dim=map_dim, reuse = True)
                        #scores.set_shape((1,map_dim))
                        scores = tf.expand_dims(scores,1)
                        #scores = tf.tile(scores,[1,3,1])
                        scores_arr.append(scores)
            return tf.convert_to_tensor(scores_arr) 
                
        if self.flag==False:
            image_feat_grid_arr0, image_feat_grid_arr1, image_feat_grid_arr2 = tf.split(input_0, num_or_size_splits=3, axis=1)
            with tf.variable_scope(self.module_variable_scope):
                with tf.variable_scope(scope, reuse=reuse):
                    H, W = self.att_shape[1:3]
                    #print("input size in count ",image_feat_grid_arr0.shape)
                    att_all = tf.reshape(image_feat_grid_arr0, to_T([-1, 1, H*W]))
                    att_min = tf.reduce_min(image_feat_grid_arr0, axis=[ 2, 3])
                    att_max = tf.reduce_max(image_feat_grid_arr0, axis=[2, 3])
                    # att_reduced has shape [N, 3]
                    #print("att shape in count ",att_all.shape, att_min.shape, att_max.shape)
                    att_concat = tf.concat([att_all, att_min, att_max], axis=2)
                    #print("shape of att_concat in count", att_concat.shape)
                    att_concat = tf.reshape(att_concat,[-1,H*W+2])
                    scores = fc('fc_scores', att_concat, output_dim=map_dim, reuse = True)
                    #print("score shape in count ",scores.shape)
                    scores = tf.expand_dims(scores,1)
                    #scores.set_shape((1,map_dim))
                    scores = tf.tile(scores,[1,3,1])
                    #print("score shape in count ",scores.shape)
            return scores
    
    def SamePropertyModule(self, input_0, input_1, time_idx, batch_idx,
        map_dim=500, scope='SamePropertyModule', reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors
        # It is only used for the layouts which have "Break"
        image_feat_grid = self._slice_image_feat_grid(batch_idx)
        image_feat_grid = tf.pad(image_feat_grid, tf.convert_to_tensor([[0,0],[0,0],[1,1],[0,0]]),'CONSTANT')
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
                    tf.nn.softmax(tf.reshape(input_0, to_T([N,3, H*W]))),
                    to_T([N,3, H, W, 1]))
                att_softmax_1 = tf.reshape(
                    tf.nn.softmax(tf.reshape(input_1, to_T([N,3, H*W]))),
                    to_T([N,3, H, W, 1]))
                image_feat_grid = tf.expand_dims(image_feat_grid,1)
                image_feat_grid = tf.tile(image_feat_grid,[1,3,1,1,1])
                # att_feat_0, att_feat_1 has shape [N, D_vis]
                att_feat_0 = tf.reduce_sum(image_feat_grid * att_softmax_0, axis=[1, 2, 3])
                att_feat_1 = tf.reduce_sum(image_feat_grid * att_softmax_1, axis=[1, 2, 3])
                att_feat_mapped_0 = tf.reshape(
                    fc('fc_att_0', att_feat_0, output_dim=map_dim),
                    to_T([N, map_dim]))
                att_feat_mapped_1 = tf.reshape(
                    fc('fc_att_1', att_feat_1, output_dim=map_dim),
                    to_T([N, map_dim]))

                eltwise_mult = tf.nn.l2_normalize(
                    att_feat_mapped_0 * text_param_mapped * att_feat_mapped_1, 1)
                scores = fc('fc_eltwise', eltwise_mult, output_dim=map_dim)
                scores=tf.expand_dims(scores,axis=1)
                scores = tf.tile(scores,[1,3,1])
                #print("score shape in same property module",scores.shape)

        return scores
   
    '''def BreakModule(self, time_idx, batch_idx, map_dim=500, scope='BreakModule',
        reuse=True):
        self.flag = True''' 
            
        
    
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
        #print("input in att_reduce:",input_0.shape)
        with tf.variable_scope(self.module_variable_scope):
            with tf.variable_scope(scope, reuse=reuse):
                att_min = tf.reduce_min(input_0, axis=[2, 3])
                att_avg = tf.reduce_mean(input_0, axis=[2, 3])
                att_max = tf.reduce_max(input_0, axis=[2, 3])
                # att_reduced has shape [3, N, 3]
                att_reduced = tf.concat([att_min, att_avg, att_max], axis=2)
                #print("att reduced shape in attReduce:",att_reduced.shape)
                #scores = fc('fc_scores', att_reduced, output_dim=map_dim)
        return att_reduced
    
    def CompareModule(self, input_0, time_idx, batch_idx, map_dim=500, scope='CompareModule',
        reuse=True):
        # In TF Fold, batch_idx and time_idx are both [N_batch, 1] tensors
        # input0 is the vector output from Count module
        print("Flag value in Compare",self.flag)
        image_feat_grid = self._slice_image_feat_grid(batch_idx)
        image_feat_grid = tf.pad(image_feat_grid, tf.convert_to_tensor([[0,0],[0,0],[1,1],[0,0]]),'CONSTANT')
        text_param = self._slice_word_vecs(time_idx, batch_idx)
        D_im = input_0.get_shape().as_list()[-1]
        D_txt = text_param.get_shape().as_list()[-1]
        with tf.variable_scope(self.module_variable_scope):
           with tf.variable_scope(scope, reuse=reuse):
               with tf.variable_scope('input_vector'):
                   W_c = tf.get_variable('weights', [D_im, map_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
                   b_c = tf.get_variable('biases', map_dim,
                    initializer=tf.constant_initializer(0.))
               with tf.variable_scope('fc_text'):
                   W_t = tf.get_variable('weights', [D_txt, map_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
                   b_t = tf.get_variable('biases', map_dim,
                    initializer=tf.constant_initializer(0.))
               with tf.variable_scope('fc_eltwise'):
                   W_e = tf.get_variable('weights', [map_dim, map_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
                   b_e = tf.get_variable('biases', map_dim,
                    initializer=tf.constant_initializer(0.))

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
        if self.flag == True:
            N = tf.shape(time_idx)[0]
            image_feat_grid_arr=[]
            image_feat_grid_arr0, image_feat_grid_arr1, image_feat_grid_arr2 = tf.split(input_0, num_or_size_splits=3, axis=1)
            image_feat_grid_arr.append(image_feat_grid_arr0)
            image_feat_grid_arr.append(image_feat_grid_arr1)
            image_feat_grid_arr.append(image_feat_grid_arr2)
            scores_arr = []
            for i in range(3):
                with tf.variable_scope(self.module_variable_scope):
                    with tf.variable_scope(scope, reuse=reuse):
                        text_param_mapped = fc('fc_text', text_param, output_dim=map_dim, reuse=True)
                        #text_param_mapped = tf.reshape(text_param_mapped, to_T([N, 1, 1, map_dim]))
                        vector_mapped = fc('input_vector',image_feat_grid_arr[i], output_dim = map_dim, reuse = True)
                        #print("vector mapped and text mapped",text_param_mapped.shape, vector_mapped.shape)
                        scores = fc('fc_eltwise', text_param_mapped + vector_mapped, output_dim=map_dim, reuse = True)
                        #scores.set_shape((1,map_dim))                    
                        scores = tf.expand_dims(scores,1)
                        #scores = tf.tile(scores,[1,3])
                        scores_arr.append(scores)
                return tf.convert_to_tensor(scores_arr) 
  
        if self.flag == False:
            image_feat_grid_arr0, image_feat_grid_arr1, image_feat_grid_arr2 = tf.split(input_0, num_or_size_splits=3, axis=1)
            with tf.variable_scope(self.module_variable_scope):
                with tf.variable_scope(scope, reuse=reuse):
                    text_param_mapped = fc('fc_text', text_param, output_dim=map_dim,reuse = True)
                    vector_mapped = fc('input_vector',image_feat_grid_arr0, output_dim = map_dim), reuse = True)
                    scores = fc('fc_eltwise', text_param_mapped + vector_mapped, output_dim=map_dim, reuse = True)
                    #scores = text_param_mapped + vector_mapped
                    scores = tf.expand_dims(scores,1)
                    #scores.set_shape((1,map_dim))
                    scores = tf.tile(scores,[1,3,1])
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
        D_im = input_0.get_shape().as_list()[-1]
        
        #D_txt = text_param.get_shape().as_list()[-1]
        with tf.variable_scope(self.module_variable_scope):
           with tf.variable_scope(scope, reuse=reuse):
               with tf.variable_scope('fc_eltwise'):
                   W_c = tf.get_variable('weights', [D_im, map_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
                   b_c = tf.get_variable('biases', map_dim,
                    initializer=tf.constant_initializer(0.))
               
        if self.flag == True:
            image_feat_grid_arr_1=[]
            image_feat_grid_arr0, image_feat_grid_arr1, image_feat_grid_arr2 = tf.split(input_0, num_or_size_splits=3, axis=1)
            image_feat_grid_arr_1.append(image_feat_grid_arr0)
            image_feat_grid_arr_1.append(image_feat_grid_arr1)
            image_feat_grid_arr_1.append(image_feat_grid_arr2)
            image_feat_grid_arr_2=[]
            image_feat_grid_arr0, image_feat_grid_arr1, image_feat_grid_arr2 = tf.split(input_1, num_or_size_splits=3, axis=1)
            image_feat_grid_arr_2.append(image_feat_grid_arr0)
            image_feat_grid_arr_2.append(image_feat_grid_arr1)
            image_feat_grid_arr_2.append(image_feat_grid_arr2)
            scores_arr = []
            for i in range(3):
                with tf.variable_scope(self.module_variable_scope):
                    with tf.variable_scope(scope, reuse=reuse):
                        scores = fc('fc_eltwise', image_feat_grid_arr_1[i] + image_feat_grid_arr_2[i], output_dim=map_dim, reuse = True)
                        scores = tf.expand_dims(scores,1)
                        #scores.set_shape((1,map_dim))
                        #scores = tf.tile(scores,[1,3])
                        scores_arr.append(scores)
            return tf.convert_to_tensor(scores_arr) 
        if self.flag == False:
            input_0_arr0,  input_0_arr1,  input_0_arr2 = tf.split(input_0, num_or_size_splits=3, axis=1)
            input_1_arr0,  input_1_arr1,  input_1_arr2 = tf.split(input_1, num_or_size_splits=3, axis=1)
            with tf.variable_scope(self.module_variable_scope):
                with tf.variable_scope(scope, reuse=reuse):
                    scores = fc('fc_eltwise', input_0_arr0 + input_1_arr0, output_dim=map_dim, reuse = True)
                    scores = tf.expand_dims(scores,1)
                    #scores.set_shape((1,map_dim))
                    scores = tf.tile(scores,[1,3,1])
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
        #print("input shape in compare att",input_0.shape,input_1.shape)
        D_im = input_0.get_shape().as_list()[-1]
        D_im2 = input_1.get_shape().as_list()[-1]
        with tf.variable_scope(self.module_variable_scope):
           with tf.variable_scope(scope, reuse=reuse):
               with tf.variable_scope('fc_att1'):
                   W_c = tf.get_variable('weights', [D_im, map_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
                   b_c = tf.get_variable('biases', map_dim,
                    initializer=tf.constant_initializer(0.))
               with tf.variable_scope('fc_att1'):
                   W_t = tf.get_variable('weights', [D_im2, map_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
                   b_t = tf.get_variable('biases', map_dim,
                    initializer=tf.constant_initializer(0.))
               with tf.variable_scope('fc_eltwise'):
                   W_e = tf.get_variable('weights', [map_dim, map_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
                   b_e = tf.get_variable('biases', map_dim,
                    initializer=tf.constant_initializer(0.))
        if self.flag == True:
            image_feat_grid_arr_1=[]
            image_feat_grid_arr0, image_feat_grid_arr1, image_feat_grid_arr2 = tf.split(input_0, num_or_size_splits=3, axis=1)
            image_feat_grid_arr_1.append(image_feat_grid_arr0[:,:,:,:14,:])
            image_feat_grid_arr_1.append(image_feat_grid_arr1[:,:,:,14:28,:])
            image_feat_grid_arr_1.append(image_feat_grid_arr2[:,:,:,28:,:])
            image_feat_grid_arr_2=[]
            image_feat_grid_arr0, image_feat_grid_arr1, image_feat_grid_arr2 = tf.split(input_1, num_or_size_splits=3, axis=1)
            image_feat_grid_arr_2.append(image_feat_grid_arr0[:,:,:,:14,:])
            image_feat_grid_arr_2.append(image_feat_grid_arr1[:,:,:,14:28,:])
            image_feat_grid_arr_2.append(image_feat_grid_arr2[:,:,:,28:,:])
            scores_arr = []
            for i in range(3):
                with tf.variable_scope(self.module_variable_scope):
                    with tf.variable_scope(scope, reuse=reuse):
                        image_feat_grid_arr_1[i]=tf.reshape(image_feat_grid_arr_1[i],[-1,10,14,1])
                        image_feat_grid_arr_2[i]=tf.reshape(image_feat_grid_arr_2[i],[-1,10,14,1])
                        N = tf.shape(time_idx)[0]
                        att_feat0_mapped = tf.reshape(
                                fc('fc_att1', image_feat_grid_arr_1[i], output_dim=map_dim, reuse = True),
                                to_T([N, map_dim]))
                        att_feat1_mapped = tf.reshape(
                                fc('fc_att2', image_feat_grid_arr_2[i], output_dim=map_dim, reuse = True),
                                to_T([N, map_dim]))
                        scores = fc('fc_eltwise', att_feat0_mapped + att_feat1_mapped, output_dim=map_dim, reuse = True)
                        scores = tf.expand_dims(scores,1)
                        #scores.set_shape((1,map_dim))
                        #scores = tf.tile(scores,[1,3])
                        scores_arr.append(scores)
                return tf.convert_to_tensor(scores_arr)
        
        if self.flag == False:
            input_0_arr0,  input_0_arr1,  input_0_arr2 = tf.split(input_0, num_or_size_splits=3, axis=1)
            input_1_arr0,  input_1_arr1,  input_1_arr2 = tf.split(input_1, num_or_size_splits=3, axis=1)
            input_0_arr0 = tf.reshape(input_0_arr0,[-1,10,42,1])
            input_1_arr0 = tf.reshape(input_1_arr0,[-1,10,42,1])
            #print(input_0_arr0.shape, input_1_arr0.shape)
            with tf.variable_scope(self.module_variable_scope):
                with tf.variable_scope(scope, reuse=reuse):
                    N = tf.shape(time_idx)[0]
                    att_feat0_mapped = tf.reshape(
                                fc('fc_att1', input_0_arr0, output_dim=map_dim,reuse = True),
                                to_T([N, map_dim]))
                    att_feat1_mapped = tf.reshape(
                                fc('fc_att2', input_1_arr0, output_dim=map_dim, reuse = True),
                                to_T([N, map_dim]))
                    scores = fc('fc_eltwise', att_feat0_mapped + att_feat1_mapped, output_dim=map_dim, reuse = True)
                    scores = tf.expand_dims(scores,1)
                    #scores.set_shape((1,map_dim))
                    scores = tf.tile(scores,[1,3,1])
            return scores
        
    
    def CombineModule(self, input_0, time_idx, batch_idx, map_dim=500, scope='CombineModule',
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
        #print("text_param shape in combine:",text_param.shape)
        #print("input size in combine:", input_0.shape)
        image_feat_grid_arr0, image_feat_grid_arr1, image_feat_grid_arr2 = tf.split(input_0, num_or_size_splits=3, axis=1)
        #image_feat_grid_arr0.set_shape((1,map_dim))
        #image_feat_grid_arr1.set_shape((1,map_dim))
        #image_feat_grid_arr2.set_shape((1,map_dim))
        #print(input_0.get_shape().as_list())
        with tf.variable_scope(self.module_variable_scope):
            with tf.variable_scope(scope, reuse=reuse):
                text_param_mapped = fc('fc_text', text_param, output_dim=3)
                #text_param_mapped = tf.reshape(text_param_mapped, to_T([N, 1, 1, 3]))
                text_param_softmax = tf.convert_to_tensor(tf.nn.softmax(text_param_mapped))
                with tf.variable_scope('token_prediction'):
                    w_y = tf.get_variable('weights', [map_dim,1],
                                  initializer=tf.contrib.layers.xavier_initializer())
                    b_y = tf.get_variable('biases', 1,
                                  initializer=tf.constant_initializer(0.))
                    #print("Image feat grid 1 shape in combine",image_feat_grid_arr0.shape)
                    image_feat_grid_arr0=tf.reshape(image_feat_grid_arr0,[-1,map_dim])
                    image_feat_grid_arr1=tf.reshape(image_feat_grid_arr1,[-1,map_dim])
                    image_feat_grid_arr2=tf.reshape(image_feat_grid_arr2,[-1,map_dim])
                    #print("image feature shape in combine:",image_feat_grid_arr1.shape)
                    input_0_mapped = tf.convert_to_tensor(tf.sigmoid(tf.nn.xw_plus_b(image_feat_grid_arr0, w_y, b_y)))
                    input_1_mapped = tf.convert_to_tensor(tf.sigmoid(tf.nn.xw_plus_b(image_feat_grid_arr1, w_y, b_y)))
                    input_2_mapped = tf.convert_to_tensor(tf.sigmoid(tf.nn.xw_plus_b(image_feat_grid_arr2, w_y, b_y)))
                    #print("input mapped shape:", input_0_mapped.shape)
                    scores_matrix = [tf.reduce_min([input_0_mapped, input_1_mapped, input_2_mapped],axis=0),
                              tf.reduce_max([input_0_mapped, input_1_mapped, input_2_mapped],axis=0),
                              tf.reduce_max([tf.reduce_min([input_0_mapped,input_1_mapped],axis=0),
                              tf.reduce_min([input_0_mapped,input_2_mapped],axis=0), 
                              tf.reduce_min([input_2_mapped,input_1_mapped],axis=0)],axis=0)]
                    a=tf.convert_to_tensor([input_0_mapped, input_1_mapped, input_2_mapped])
                    #print(a.shape)
                    scores_matrix = tf.convert_to_tensor(scores_matrix)
                    #print("scores_matrix shape:",scores_matrix.shape)
                    scores_matrix = tf.reshape(tf.transpose(scores_matrix,[1,0,2]),[-1,3])
                    #print("Scores matrix shape:",scores_matrix.shape)
                    #score = tf.matmul(text_param_softmax,tf.matrix_transpose(scores_matrix))
                    #score = tf.tensordot(text_param_softmax,scores_matrix,axes=1)
                    score = tf.reduce_sum(tf.convert_to_tensor(text_param_softmax*scores_matrix),axis=1)
                    scores = tf.transpose(tf.convert_to_tensor([score,1-score]),[1,0])
                    #print("score shape:",score.shape)
                    #print("Combine scores shape",scores.shape)
                #scores = tf.convert_to_tensor([0,1])
            self.flag = False
        return scores
    
    def ExistAttModule(self, input_0, time_idx, batch_idx, map_dim=500, scope='ExistAttModule',
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
        
        image_feat_grid_arr0, image_feat_grid_arr1, image_feat_grid_arr2 = tf.split(input_0, num_or_size_splits=3, axis=1)
        with tf.variable_scope(self.module_variable_scope):
            with tf.variable_scope(scope, reuse=reuse):
                N = tf.shape(time_idx)[0]
                text_param = self._slice_word_vecs(time_idx, batch_idx)
                text_param_mapped = fc('fc_text', text_param, output_dim=3)
                text_param_softmax = tf.nn.softmax(text_param_mapped)
                with tf.variable_scope('token_prediction'):
                    w_y = tf.get_variable('weights', [3,1],
                       initializer=tf.contrib.layers.xavier_initializer())
                    b_y = tf.get_variable('biases', 1,
                    initializer=tf.constant_initializer(0.))
                    image_feat_grid_arr0=tf.reshape(image_feat_grid_arr0,[-1,3])
                    image_feat_grid_arr1=tf.reshape(image_feat_grid_arr1,[-1,3])
                    image_feat_grid_arr2=tf.reshape(image_feat_grid_arr2,[-1,3])
                    input_0_mapped = tf.convert_to_tensor(tf.sigmoid(tf.nn.xw_plus_b(image_feat_grid_arr0, w_y, b_y)))
                    input_1_mapped = tf.convert_to_tensor(tf.sigmoid(tf.nn.xw_plus_b(image_feat_grid_arr1, w_y, b_y)))
                    input_2_mapped = tf.convert_to_tensor(tf.sigmoid(tf.nn.xw_plus_b(image_feat_grid_arr2, w_y, b_y)))
                    scores_matrix = [tf.reduce_min([input_0_mapped, input_1_mapped, input_2_mapped],axis=0),
                              tf.reduce_max([input_0_mapped, input_1_mapped, input_2_mapped],axis=0),
                              tf.reduce_max([tf.reduce_min([input_0_mapped,input_1_mapped],axis=0),
                              tf.reduce_min([input_0_mapped,input_2_mapped],axis=0), 
                              tf.reduce_min([input_2_mapped,input_1_mapped],axis=0)],axis=0)]
                    scores_matrix = tf.convert_to_tensor(scores_matrix)
                    #scores_matrix = tf.convert_to_tensor(scores_matrix)
                    #print("scores_matrix shape:",scores_matrix.shape)
                    scores_matrix = tf.reshape(tf.transpose(scores_matrix,[1,0,2]),[-1,3])
                    #print("Scores matrix shape:",scores_matrix.shape)
                    #score = tf.matmul(text_param_softmax,tf.matrix_transpose(scores_matrix))
                    #score = tf.tensordot(text_param_softmax,scores_matrix,axes=1)
                    score = tf.reduce_sum(tf.convert_to_tensor(text_param_softmax*scores_matrix),axis=1)
                    a = tf.convert_to_tensor([score,1-score])
                    #print("ExistAtt shape ",a.shape)
                    scores = tf.transpose(tf.convert_to_tensor([score,1-score]),[1,0])
                    #print("score shape:",score.shape)
                    #print("Exist Att scores shape",scores.shape)
                    #scores = tf.convert_to_tensor([0,1])

                    #scores_matrix = tf.reshape(scores_matrix,[3,1])
                    #score = tf.matmul(text_param_softmax,scores_matrix)
                    #score = tf.matmul(text_param_softmax,tf.matrix_transpose(scores_matrix))
                    #scores = tf.convert_to_tensor([score,1-score])
                    #scores = fc('fc_scores', scores, output_dim=self.num_choices)
                    #scores = tf.nn.softmax(scores)
            #scores = tf.convert_to_tensor([0,1])
            self.flag = False
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
        with tf.variable_scope(self.module_variable_scope):
            with tf.variable_scope(scope, reuse=reuse):
                scores = fc('fc_scores', input_0, output_dim=self.num_choices)
                scores = tf.nn.softmax(scores)
                self.flag = False
        return scores
