from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, Add, MaxPooling3D, Activation, BatchNormalization, UpSampling3D
from keras.layers import Conv3DTranspose
from keras.initializers import RandomNormal

def Transition():
    pass

def DenseBlock():
    pass

def BottleNeck():
    pass

def LastConv():
    pass

def FirstConv():
    pass

# First Conv Parameters
num_init_filters = 8

# DenseBlock Parameters
bottleneck_factor   = 4 # no. of output feature maps for 1x1 Conv
growth_rate         = 4 # no. of output feature maps per Conv. Layer in DenseBlocks

# Transition Layer Parameters
theta               = 0.5 # to determine number of output feature maps of a DenseBlock

# NOTE:
# Dilated convolutions happen in the encoder part

def DenseNet(input):
    # First convolution
    x = Conv3D(num_init_filters, kernel_size=5, strides=2, dilation_rate=1, padding='same', name='conv0', activation='relu')(input) # (240, 144, 240)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(x) # (120,72,120)

    # DenseBlock_1 (4 layers)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(x)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer1  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer1  = BatchNormalization()(out_layer1)
    out_layer1  = Activation('relu')(out_layer1)

    input_cat   = concatenate([x, out_layer1],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer2  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer2  = BatchNormalization()(out_layer2)
    out_layer2  = Activation('relu')(out_layer2)

    input_cat   = concatenate([x, out_layer1, out_layer2],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer3  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer3  = BatchNormalization()(out_layer3)
    out_layer3  = Activation('relu')(out_layer3)

    input_cat   = concatenate([x, out_layer1, out_layer2, out_layer3],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer4  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer4  = BatchNormalization()(out_layer4)
    out_layer4  = Activation('relu')(out_layer4)

    # check for shape of feature map
    # Transition_1 (in:(120,72,120) -> out:(60,36,60))
    input_cat   = concatenate([x, out_layer1, out_layer2, out_layer3, out_layer4],axis=-1)
    out_trans   = Conv3D(theta * input_cat.shape[0], kernel_size=1, strides=1, padding='valid')(in_trans)
    out_trans   = BatchNormalization()(out_trans)
    out_trans   = Activation('relu')(out_trans)
    # out_trans   = AveragePooling3D(pool_size=(2,2,2), strides=(2,2,2))
    out_trans   = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))

    # DenseBlock_2 (8 layers)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(out_trans)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer1  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer1  = BatchNormalization()(out_layer1)
    out_layer1  = Activation('relu')(out_layer1)

    input_cat   = concatenate([out_trans, out_layer1],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer2  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer2  = BatchNormalization()(out_layer2)
    out_layer2  = Activation('relu')(out_layer2)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer3  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer3  = BatchNormalization()(out_layer3)
    out_layer3  = Activation('relu')(out_layer3)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer4  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer4  = BatchNormalization()(out_layer4)
    out_layer4  = Activation('relu')(out_layer4)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer5  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer5  = BatchNormalization()(out_layer5)
    out_layer5  = Activation('relu')(out_layer5)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer6  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer6  = BatchNormalization()(out_layer6)
    out_layer6  = Activation('relu')(out_layer6)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6],axis=-1)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    out_layer7  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer7  = BatchNormalization()(out_layer7)
    out_layer7  = Activation('relu')(out_layer7)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer8  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer8  = BatchNormalization()(out_layer8)
    out_layer8  = Activation('relu')(out_layer8)

    # Transition_2 (in:(60,36,60) -> out:(30,18,30))
    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7, out_layer8],axis=-1)
    out_trans   = Conv3D(theta * input_cat.shape[0], kernel_size=1, strides=1, padding='valid')(input_cat)
    out_trans   = BatchNormalization()(out_trans)
    out_trans   = Activation('relu')(out_trans)
    # out_trans   = AveragePooling3D(pool_size=(2,2,2), strides=(2,2,2))
    out_trans   = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))

    # DenseBlock_3 (16 layers)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(out_trans)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer1  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer1  = BatchNormalization()(out_layer1)
    out_layer1  = Activation('relu')(out_layer1)

    input_cat   = concatenate([out_trans, out_layer1],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer2  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer2  = BatchNormalization()(out_layer2)
    out_layer2  = Activation('relu')(out_layer2)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer3  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer3  = BatchNormalization()(out_layer3)
    out_layer3  = Activation('relu')(out_layer3)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer4  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer4  = BatchNormalization()(out_layer4)
    out_layer4  = Activation('relu')(out_layer4)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer5  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer5  = BatchNormalization()(out_layer5)
    out_layer5  = Activation('relu')(out_layer5)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer6  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer6  = BatchNormalization()(out_layer6)
    out_layer6  = Activation('relu')(out_layer6)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer7  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer7  = BatchNormalization()(out_layer7)
    out_layer7  = Activation('relu')(out_layer7)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer8  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer8  = BatchNormalization()(out_layer8)
    out_layer8  = Activation('relu')(out_layer8)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7,
                                out_layer8],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer9  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer9  = BatchNormalization()(out_layer9)
    out_layer9  = Activation('relu')(out_layer9)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7,
                                out_layer8, out_layer9],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer10 = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer10 = BatchNormalization()(out_layer10)
    out_layer10 = Activation('relu')(out_layer10)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7,
                                out_layer8, out_layer9, out_layer10],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer11 = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer11 = BatchNormalization()(out_layer11)
    out_layer11 = Activation('relu')(out_layer11)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7,
                                out_layer8, out_layer9, out_layer10, out_layer11],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer12 = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer12 = BatchNormalization()(out_layer12)
    out_layer12 = Activation('relu')(out_layer12)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7,
                                out_layer8, out_layer9, out_layer10, out_layer11, out_layer12],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer13 = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer13 = BatchNormalization()(out_layer13)
    out_layer13 = Activation('relu')(out_layer13)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7,
                                out_layer8, out_layer9, out_layer10, out_layer11, out_layer12, out_layer13],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer14 = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer14 = BatchNormalization()(out_layer14)
    out_layer14 = Activation('relu')(out_layer14)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7,
                                out_layer8, out_layer9, out_layer10, out_layer11, out_layer12, out_layer13, out_layer14],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer15 = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer15 = BatchNormalization()(out_layer15)
    out_layer15 = Activation('relu')(out_layer15)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7,
                                out_layer8, out_layer9, out_layer10, out_layer11, out_layer12, out_layer13, out_layer14, out_layer15],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid')(input_cat)
    bottleneck  = BatchNormalization()(bottleneck)
    bottleneck  = Activation('relu')(bottleneck)
    out_layer16 = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same')(bottleneck)
    out_layer16 = BatchNormalization()(out_layer16)
    out_layer16 = Activation('relu')(out_layer16)
    # do batchnorm + relu here

    # Upsampling from (30,18,30) -> (60,36,60)
    ### Which output is being passed from previous DenseBlock?
    # 1. The concatenation of all the previous feature maps or
    # 2. Only the output feature map of the last layer within the DenseBlock
    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7,
                                out_layer8, out_layer9, out_layer10, out_layer11, out_layer12, out_layer13, out_layer14, out_layer15, out_layer16],axis=-1)
    input_cat   = BatchNormalization()(input_cat)
    input_cat   = Activation('relu')(input_cat)
    up_1 = Conv3DTranspose(64, kernel_size=(2, 2, 2), strides=(2,2,2))(input_cat)

    # what about
    # up_1 = Conv3DTranspose(64, kernel_size=(3,3,3), strides=(1,1,1))(out_layer16)
    up_1    = Conv3D(32, kernel_size=3, strides=1, padding='same')(up_1)
    up_1    = BatchNormalization()(up_1)
    up_1    = Activation('relu')(up_1)
    up_1    = Conv3D(16, kernel_size=1, strides=1, padding='same')(up_1)
    up_1    = Conv3D(16, kernel_size=1, strides=1, padding='same')(up_1)
    final   = Conv3D(12, kernel_size=1, strides=1, padding='same')(up_1)

    return final


def get_densenet():
    input_tsdf = Input(shape=(240, 144, 240, 1))
    fin = DenseNet(x)
    model = Model(inputs=input_tsdf, outputs=fin)
    return model

def get_network_by_name(name):
    if name == 'DENSENET':
        return get_densenet(), 'depth'

def get_net_name_from_w(weights):
    networks = ['DENSENET']

    for net in networks:
        if ((net+'_LR') == (weights[0:len(net)+3])) or ((net+'fine_LR') == (weights[0:len(net)+7])):
            return net

    print("Invalid weight file: ", weights)
    exit(-1)
