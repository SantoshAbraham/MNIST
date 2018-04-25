'''
Python code using tensor flow for the MNIST data set
The needed network is fed in as dictionary called "NetworkArchitecture"
The names of the variables are self explanatory
You can change the descriptions of the layers by
modifying the parameters in NetworkArchitecture
'''

import numpy as np
import tensorflow as tf

#Read in Mnist Data
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("./MNIST/", one_hot=True)




#Model Definition and Hyper Parameters
NumberOfCategories = 10 #Number of output categoris 
InputPictureDimensions = [28, 28, 1]

#Use the following to have a convolutional network
NetworkArchitecture = {
    'LearningRate' : 0.001,
    'NumberOfLayers': 5,
    'NumberOfCategories': NumberOfCategories,
    'layer_0_type': 'Input',
    'layer_0_InputShape': InputPictureDimensions,   
    #
    'layer_1_type': 'Conv',
    'layer_1_kernel_size': 5,
    'layer_1_NumFilters': 32,
    'layer_1_PoolSize': 2,
    'layer_1_Padding': 0,
    'layer_1_ConvStride': 1,
    'layer_1_PoolStride': 2,
    'layer_1_Activation': tf.nn.relu,
    #
    'layer_2_type': 'Conv',
    'layer_2_kernel_size': 5,
    'layer_2_NumFilters': 64,
    'layer_2_PoolSize': 2,
    'layer_2_Padding': 0,
    'layer_2_ConvStride': 1,
    'layer_2_PoolStride': 2,
    'layer_2_Activation': tf.nn.relu,    
    #
    'layer_3_type': 'Dense',
    'layer_3_Activation': tf.nn.relu,
    'layer_3_size': 1024,
    'layer_3_dropout': 0.4,
    #
    'layer_4_type': 'Dense',
    'layer_4_Activation': None,
    'layer_4_size': 10,
    'layer_4_dropout': 0    
}

#Use the following to have a network with layers that are only dense
'''
NetworkArchitecture = {
    'LearningRate' : 0.001,
    'NumberOfLayers': 4,
    #
    'NumberOfCategories': NumberOfCategories,
    'layer_0_type': 'Input',
    'layer_0_InputShape': InputPictureDimensions,        
    #
    'layer_1_type': 'Dense',
    'layer_1_Activation': tf.nn.relu,
    'layer_1_size': 100,
    'layer_1_dropout': 0.1, 
    #
    'layer_2_type': 'Dense',
    'layer_2_Activation': tf.nn.relu,
    'layer_2_size': 100,
    'layer_2_dropout': 0.1,
    #
    'layer_3_type': 'Dense',
    'layer_3_Activation': None,
    'layer_3_size': NumberOfCategories,
    'layer_3_dropout': 0    
}
'''
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#This function computes the output size of a convolutional layer if one
#is there in NetworkArchitecture
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
def populate_conv_output_sizes(NWArch):
    NumLayers = NWArch['NumberOfLayers']
    
    for i in range(1,NumLayers):
        PreFix = 'layer_'+str(i)+'_'
        PrevPreFix = 'layer_'+str(i-1)+'_'
        
        if NWArch[PreFix+'type'] != 'Conv':
            continue
        
        if NWArch[PrevPreFix + 'type'] != 'Dense':
            rows = NWArch[PrevPreFix+'size'][0]
            cols = NWArch[PrevPreFix+'size'][1]
            chans = NWArch[PrevPreFix+'size'][2]

        
        
        padding = NWArch[PreFix + 'Padding']
        conv_stride = NWArch[PreFix + 'ConvStride']
        kernel_size = NWArch[PreFix + 'kernel_size']
        numFilters = NWArch[PreFix + 'NumFilters']
        pool_stride = NWArch[PreFix + 'PoolStride']
        pool_size = NWArch[PreFix + 'PoolSize']

        conv_out_rows = rows #(rows - kernel_size + 2*padding+conv_stride)/(conv_stride )
        conv_out_cols = cols #(cols - kernel_size + 2*padding+conv_stride)/(conv_stride )

        conv_out_rows = int((conv_out_rows - pool_size + 2*padding+pool_stride)/(pool_stride ))
        conv_out_cols = int((conv_out_cols - pool_size + 2*padding+pool_stride)/(pool_stride )     )      

        NWArch[PreFix + 'size'] = ([conv_out_rows, conv_out_cols, numFilters])
        
        print(i, NWArch[PreFix+'size'])


        

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#This function creates convolutional layer given the layer number and
#the corresponding layer parameters e.g. number of outputs, activation etc.
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def create_layer_conv(layer_number,NWArch):
    '''
    layer_number - Number of the Layer
    NWArch:  Dictionary of the architecutre, layer_number
    indexes into the NWArch dictionary
    '''
    PreFix = 'layer_'+str(layer_number)+'_'
    PrevPreFix = 'layer_'+str(layer_number-1)+'_'

    #Sanity Check
    if NWArch[PreFix + 'type'] != 'Conv':
        print('Incompatible layer number and Conv')
        return None
    
    conv1 = tf.layers.conv2d(
        inputs = NWArch[PrevPreFix + 'output'],
        filters = NWArch[PreFix + 'NumFilters'],
        kernel_size = NWArch[PreFix + 'kernel_size'],
        padding = "same",
        activation = NWArch[PreFix + 'Activation'])
    
    NWArch[PreFix + 'output']= tf.layers.max_pooling2d(
        inputs = conv1,
        pool_size = NWArch[PreFix + 'PoolSize'],
        strides = NWArch[PreFix + 'PoolStride'])
    
    print(layer_number, NWArch[PreFix+'size'])



# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#This function creates a dense neural layer given the layer number and
#the corresponding layer parameters e.g. number of neurons, activation
#dropout etc.
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


        
def create_layer_dense(layer_number,NWArch):
    '''
    layer_number - Number of the Layer
    NWArch:  Dictionary of the architecutre, layer_number
    indexes into the NWArch dictionary
    '''
    
    PreFix = 'layer_'+str(layer_number)+'_'
    PrevPreFix = 'layer_'+str(layer_number-1)+'_'
    
    #Sanity Check
    if NWArch[PreFix + 'type'] != 'Dense':
        print('Incompatible layer number and Dense')
        return None
    
    if NWArch[PrevPreFix + 'type'] == 'Conv':        
        prevOutSize = NWArch[PrevPreFix + 'size'][0] *             NWArch[PrevPreFix + 'size'][1] * NWArch[PrevPreFix + 'size'][2]
        InputTensor = tf.reshape(NWArch[PrevPreFix + 'output'],[-1,prevOutSize])
        print('reshape',layer_number,prevOutSize,NWArch[PrevPreFix + 'output'].shape)
            
    if NWArch[PrevPreFix + 'type'] == 'Dense':
        prevOutSize = NWArch[PrevPreFix + 'size']
        InputTensor = NWArch[PrevPreFix + 'output']
        print(layer_number,prevOutSize)
        
    if NWArch[PrevPreFix + 'type'] == 'Input':
        prevOutSize = NWArch[PrevPreFix + 'size']
        InputTensor = tf.reshape(NWArch[PrevPreFix + 'output'],[-1,prevOutSize])
        print('reshape',layer_number,prevOutSize,NWArch[PrevPreFix + 'output'].shape)
    
    DenseOutput = tf.layers.dense(
        inputs = InputTensor,                            
        units = NWArch[PreFix + 'size'],
        activation = NWArch[PreFix + 'Activation'])
    
    if NWArch['NumberOfLayers'] == layer_number + 1:
        NWArch[PreFix + 'output'] = tf.layers.dropout(
            inputs = DenseOutput,
            rate = NWArch[PreFix + 'dropout'])
    else:
        NWArch[PreFix+'output'] = DenseOutput
    
    print(NWArch[PreFix + 'output'].shape)


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#This function creates the input layer to the network
# If the first hidden layer is a convolutional layer, then this layer outputs
#a 3 dimensional array (x,y, and number of channels
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
def create_input_output_layer(NWArch):
    PreFix = 'layer_0_' #Hardcoding layer 0 as input layer
    NextLayer = 'layer_1_'
    '''
    The main reason to write a separate function is that
    if the first layer is convolutional then we want a 2 dimensional array
    to represent the pictures  other wise we want the input to be
    one dimensional'''
    
    #Sanity check
    if NWArch[PreFix+'type'] != 'Input':        
        print('Layer 0 is not input ERROR')
        return None
    
    PictShape = NWArch[PreFix + 'InputShape'] 
    PictureSizeSqueezed = PictShape[0] * PictShape[1] * PictShape[2]
    
    #This code is particular to MNIST since it 
    NWArch[PreFix+'input'] = tf.placeholder(tf.float32,[None,PictureSizeSqueezed])    
    NWArch['OutPut'] = tf.placeholder(tf.float32,[None,NumberOfCategories])
    
    if NWArch[NextLayer+'type'] == 'Conv':
        NWArch[PreFix+'output'] = tf.reshape(NWArch[PreFix+'input'],[-1,PictShape[0],PictShape[1],PictShape[2] ])
        NWArch[PreFix+'size'] = NWArch[PreFix+'InputShape']
        return 
    
    if NWArch[NextLayer+'type'] == 'Dense':        
        NWArch[PreFix+'output'] = NWArch[PreFix+'input']
        NWArch[PreFix+'size'] = PictureSizeSqueezed
        return 
    
        

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#Step 1:  Bulild the network as specified by the NetworkArchitecture Dictionary
#Note the built layers are added to the NetworkArchitecture dictionary
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
create_input_output_layer(NetworkArchitecture)
populate_conv_output_sizes(NetworkArchitecture)

for i in range(1,NetworkArchitecture['NumberOfLayers']):
    
    PreFix = 'layer_'+str(i)+'_'
        
    if NetworkArchitecture[PreFix+'type'] == 'Conv':
        create_layer_conv(i,NetworkArchitecture)
    
    
    if NetworkArchitecture[PreFix+'type'] == 'Dense':
        create_layer_dense(i,NetworkArchitecture)


#Define th eloss function 		
o_name = 'layer_'+str(NetworkArchitecture['NumberOfLayers'] - 1)+'_output'
NetworkArchitecture['loss'] =     tf.nn.softmax_cross_entropy_with_logits(logits=NetworkArchitecture[o_name],labels=NetworkArchitecture['OutPut']) 
 
out_equals_target = tf.equal(tf.argmax(NetworkArchitecture[o_name], 1), tf.argmax(NetworkArchitecture['OutPut'], 1))

accuracy = tf.reduce_mean(tf.cast(out_equals_target, tf.float32))

OverallLoss = tf.reduce_mean(NetworkArchitecture['loss'])

#Define the optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=NetworkArchitecture['LearningRate']).minimize(OverallLoss)

#Create the Session and run the optimizer
sess = tf.Session()
initializer = tf.global_variables_initializer()
sess.run(initializer)
for k in range(1000):
    for i in range(10):
        batch_size = 500
        input_batch, target_batch = data.train.next_batch(batch_size)
        for j in range(1):
            _,x= sess.run([optimizer,OverallLoss], feed_dict={NetworkArchitecture['layer_0_input']: input_batch, 
                                                              NetworkArchitecture['OutPut']:target_batch})    
        print(k,i,j,x)




#Check the network on the validation set
input_batch, target_batch = data.validation.next_batch(data.train._num_examples)
validation_loss, validation_accuracy = sess.run([OverallLoss, accuracy], 
    feed_dict={NetworkArchitecture['layer_0_input']: input_batch, 
                     NetworkArchitecture['OutPut']:target_batch}) 
print(validation_loss,validation_accuracy)

#Check the network on the t set
input_batch, target_batch = data.test.next_batch(data.test._num_examples)
validation_loss, validation_accuracy = sess.run([OverallLoss, accuracy], 
    feed_dict={NetworkArchitecture['layer_0_input']: input_batch, 
                     NetworkArchitecture['OutPut']:target_batch}) 
print(validation_loss,validation_accuracy)






