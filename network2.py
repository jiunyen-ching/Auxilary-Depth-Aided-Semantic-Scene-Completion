from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, Add, MaxPooling3D, Activation, BatchNormalization, UpSampling3D, Dropout
from keras.layers import Conv3DTranspose
from keras.initializers import RandomNormal

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
    x = Conv3D(num_init_filters, kernel_size=3, strides=1, padding='same', name='FirstConv_Conv1')(input) # (240, 144, 240, 8)
    x = BatchNormalization(name='FirstConv_BN1')(x)
    x = Activation('relu', name='FirstConv_Act1')(x)
    x = Conv3D(num_init_filters, kernel_size=3, strides=1, padding='same', name='FirstConv_Conv2')(x) # (240, 144, 240, 8)
    x = BatchNormalization(name='FirstConv_BN2')(x)
    x = Activation('relu', name='FirstConv_Act2')(x)
    x = Conv3D(num_init_filters, kernel_size=3, strides=1, padding='same', name='FirstConv_Conv3')(x) # (240, 144, 240, 8)
    x = BatchNormalization(name='FirstConv_BN3')(x)
    x = Activation('relu', name='FirstConv_Act3')(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), name='MaxPool_1')(x) # (120,72,120, 8)

    # DenseBlock_1 (4 layers)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB1_BotNeck1_Conv')(x)
    bottleneck  = BatchNormalization(name='DB1_BotNeck1_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB1_BotNeck1_Act')(bottleneck)
    out_layer1  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB1_Layer1_Conv')(bottleneck)
    out_layer1  = BatchNormalization(name='DB1_Layer1_BN')(out_layer1)
    out_layer1  = Activation('relu', name='DB1_Layer1_Act')(out_layer1)

    input_cat   = concatenate([x, out_layer1],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB1_BotNeck2_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB1_BotNeck2_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB1_BotNeck2_Act')(bottleneck)
    out_layer2  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB1_Layer2_Conv')(bottleneck)
    out_layer2  = BatchNormalization(name='DB1_Layer2_BN')(out_layer2)
    out_layer2  = Activation('relu', name='DB1_Layer2_Act')(out_layer2)

    input_cat   = concatenate([x, out_layer1, out_layer2],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB1_BotNeck3_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB1_BotNeck3_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB1_BotNeck3_Act')(bottleneck)
    out_layer3  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB1_Layer3_Conv')(bottleneck)
    out_layer3  = BatchNormalization(name='DB1_Layer3_BN')(out_layer3)
    out_layer3  = Activation('relu', name='DB1_Layer3_Act')(out_layer3)

    input_cat   = concatenate([x, out_layer1, out_layer2, out_layer3],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB1_BotNeck4_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB1_BotNeck4_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB1_BotNeck4_Act')(bottleneck)
    out_layer4  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB1_Layer4_Conv')(bottleneck)
    out_layer4  = BatchNormalization(name='DB1_Layer4_BN')(out_layer4)
    out_layer4  = Activation('relu', name='DB1_Layer4_Act')(out_layer4)
    out_layer4  = Dropout(rate=0.2)(out_layer4)

    # check for shape of feature map
    # Transition_1 (in:(120,72,120) -> out:(60,36,60))
    input_cat   = concatenate([x, out_layer1, out_layer2, out_layer3, out_layer4],axis=-1)
    out_trans   = Conv3D(int(theta * int(input_cat.shape[-1])), kernel_size=1, strides=1, padding='valid', name='Transition1_Conv')(input_cat)
    out_trans   = BatchNormalization(name='Transition1_BN')(out_trans)
    out_trans   = Activation('relu', name='Transition1_Act')(out_trans)
    # out_trans   = AveragePooling3D(pool_size=(2,2,2), strides=(2,2,2))(out_trans)
    out_trans   = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), name='Transition1_MaxPool')(out_trans)

    # DenseBlock_2 (8 layers)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB2_BotNeck1_Conv')(out_trans)
    bottleneck  = BatchNormalization(name='DB2_BotNeck1_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB2_BotNeck1_Act')(bottleneck)
    out_layer1  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB2_Layer1_Conv')(bottleneck)
    out_layer1  = BatchNormalization(name='DB2_Layer1_BN')(out_layer1)
    out_layer1  = Activation('relu', name='DB2_Layer1_Act')(out_layer1)

    input_cat   = concatenate([out_trans, out_layer1],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB2_BotNeck2_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB2_BotNeck2_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB2_BotNeck2_Act')(bottleneck)
    out_layer2  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB1_Layer2_Conv')(bottleneck)
    out_layer2  = BatchNormalization(name='DB2_Layer2_BN')(out_layer2)
    out_layer2  = Activation('relu', name='DB2_Layer2_Act')(out_layer2)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB2_BotNeck3_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB2_BotNeck3_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB2_BotNeck3_Act')(bottleneck)
    out_layer3  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB2_Layer3_Conv')(bottleneck)
    out_layer3  = BatchNormalization(name='DB2_Layer3_BN')(out_layer3)
    out_layer3  = Activation('relu', name='DB2_Layer3_Act')(out_layer3)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB2_BotNeck4_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB2_BotNeck4_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB2_BotNeck4_Act')(bottleneck)
    out_layer4  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB2_Layer4_Conv')(bottleneck)
    out_layer4  = BatchNormalization(name='DB2_Layer4_BN')(out_layer4)
    out_layer4  = Activation('relu', name='DB2_Layer4_Act')(out_layer4)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB2_BotNeck5_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB2_BotNeck5_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB2_BotNeck5_Act')(bottleneck)
    out_layer5  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB2_Layer5_Conv')(bottleneck)
    out_layer5  = BatchNormalization(name='DB2_Layer5_BN')(out_layer5)
    out_layer5  = Activation('relu', name='DB2_Layer5_Act')(out_layer5)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB2_BotNeck6_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB2_BotNeck6_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB2_BotNeck6_Act')(bottleneck)
    out_layer6  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB2_Layer6_Conv')(bottleneck)
    out_layer6  = BatchNormalization(name='DB2_Layer6_BN')(out_layer6)
    out_layer6  = Activation('relu', name='DB2_Layer6_Act')(out_layer6)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6],axis=-1)
    bottleneck  = BatchNormalization(name='DB2_BotNeck7_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB2_BotNeck7_Act')(bottleneck)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB2_BotNeck7_Conv')(input_cat)
    out_layer7  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB2_Layer7_Conv')(bottleneck)
    out_layer7  = BatchNormalization(name='DB2_Layer7_BN')(out_layer7)
    out_layer7  = Activation('relu', name='DB2_Layer7_Act')(out_layer7)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB2_BotNeck8_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB2_BotNeck8_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB2_BotNeck8_Act')(bottleneck)
    out_layer8  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB2_Layer8_Conv')(bottleneck)
    out_layer8  = BatchNormalization(name='DB2_Layer8_BN')(out_layer8)
    out_layer8  = Activation('relu', name='DB2_Layer8_Act')(out_layer8)
    out_layer8  = Dropout(rate=0.2)(out_layer8)

    # Transition_2 (in:(60,36,60) -> out:(30,18,30))
    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7, out_layer8],axis=-1)
    out_trans   = Conv3D(int(theta * int(input_cat.shape[-1])), kernel_size=1, strides=1, padding='valid', name='Transition2_Conv')(input_cat)
    out_trans   = BatchNormalization(name='Transition2_BN')(out_trans)
    out_trans   = Activation('relu', name='Transition2_Act')(out_trans)
    # out_trans   = AveragePooling3D(pool_size=(2,2,2), strides=(2,2,2))(out_trans)
    out_trans   = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), name='Transition2_MaxPool')(out_trans)

    # DenseBlock_3 (16 layers)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB3_BotNeck1_Conv')(out_trans)
    bottleneck  = BatchNormalization(name='DB3_BotNeck1_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB3_BotNeck1_Act')(bottleneck)
    out_layer1  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB3_Layer1_Conv')(bottleneck)
    out_layer1  = BatchNormalization(name='DB3_Layer1_BN')(out_layer1)
    out_layer1  = Activation('relu', name='DB3_Layer1_Act')(out_layer1)

    input_cat   = concatenate([out_trans, out_layer1],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB3_BotNeck2_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB3_BotNeck2_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB3_BotNeck2_Act')(bottleneck)
    out_layer2  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB3_Layer2_Conv')(bottleneck)
    out_layer2  = BatchNormalization(name='DB3_Layer2_BN')(out_layer2)
    out_layer2  = Activation('relu', name='DB3_Layer2_Act')(out_layer2)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB3_BotNeck3_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB3_BotNeck3_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB3_BotNeck3_Act')(bottleneck)
    out_layer3  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB3_Layer3_Conv')(bottleneck)
    out_layer3  = BatchNormalization(name='DB3_Layer3_BN')(out_layer3)
    out_layer3  = Activation('relu', name='DB3_Layer3_Act')(out_layer3)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB3_BotNeck4_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB3_BotNeck4_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB3_BotNeck4_Act')(bottleneck)
    out_layer4  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB3_Layer4_Conv')(bottleneck)
    out_layer4  = BatchNormalization(name='DB3_Layer4_BN')(out_layer4)
    out_layer4  = Activation('relu', name='DB3_Layer4_Act')(out_layer4)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB3_BotNeck5_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB3_BotNeck5_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB3_BotNeck5_Act')(bottleneck)
    out_layer5  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB3_Layer5_Conv')(bottleneck)
    out_layer5  = BatchNormalization(name='DB3_Layer5_BN')(out_layer5)
    out_layer5  = Activation('relu', name='DB3_Layer5_Act')(out_layer5)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB3_BotNeck6_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB3_BotNeck6_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB3_BotNeck6_Act')(bottleneck)
    out_layer6  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB3_Layer6_Conv')(bottleneck)
    out_layer6  = BatchNormalization(name='DB3_Layer6_BN')(out_layer6)
    out_layer6  = Activation('relu', name='DB3_Layer6_Act')(out_layer6)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB3_BotNeck7_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB3_BotNeck7_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB3_BotNeck7_Act')(bottleneck)
    out_layer7  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB3_Layer7_Conv')(bottleneck)
    out_layer7  = BatchNormalization(name='DB3_Layer7_BN')(out_layer7)
    out_layer7  = Activation('relu', name='DB3_Layer7_Act')(out_layer7)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB3_BotNeck8_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB3_BotNeck8_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB3_BotNeck8_Act')(bottleneck)
    out_layer8  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB3_Layer8_Conv')(bottleneck)
    out_layer8  = BatchNormalization(name='DB3_Layer8_BN')(out_layer8)
    out_layer8  = Activation('relu', name='DB3_Layer8_Act')(out_layer8)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7,
                                out_layer8],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB3_BotNeck9_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB3_BotNeck9_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB3_BotNeck9_Act')(bottleneck)
    out_layer9  = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB3_Layer9_Conv')(bottleneck)
    out_layer9  = BatchNormalization(name='DB3_Layer9_BN')(out_layer9)
    out_layer9  = Activation('relu', name='DB3_Layer9_Act')(out_layer9)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7,
                                out_layer8, out_layer9],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB3_BotNeck10_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB3_BotNeck10_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB3_BotNeck10_Act')(bottleneck)
    out_layer10 = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB3_Layer10_Conv')(bottleneck)
    out_layer10 = BatchNormalization(name='DB3_Layer10_BN')(out_layer10)
    out_layer10 = Activation('relu', name='DB3_Layer10_Act')(out_layer10)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7,
                                out_layer8, out_layer9, out_layer10],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB3_BotNeck11_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB3_BotNeck11_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB3_BotNeck11_Act')(bottleneck)
    out_layer11 = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB3_Layer11_Conv')(bottleneck)
    out_layer11 = BatchNormalization(name='DB3_Layer11_BN')(out_layer11)
    out_layer11 = Activation('relu', name='DB3_Layer11_Act')(out_layer11)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7,
                                out_layer8, out_layer9, out_layer10, out_layer11],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB3_BotNeck12_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB3_BotNeck12_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB3_BotNeck12_Act')(bottleneck)
    out_layer12 = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB3_Layer12_Conv')(bottleneck)
    out_layer12 = BatchNormalization(name='DB3_Layer12_BN')(out_layer12)
    out_layer12 = Activation('relu', name='DB3_Layer12_Act')(out_layer12)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7,
                                out_layer8, out_layer9, out_layer10, out_layer11, out_layer12],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB3_BotNeck13_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB3_BotNeck13_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB3_BotNeck13_Act')(bottleneck)
    out_layer13 = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB3_Layer13_Conv')(bottleneck)
    out_layer13 = BatchNormalization(name='DB3_Layer13_BN')(out_layer13)
    out_layer13 = Activation('relu', name='DB3_Layer13_Act')(out_layer13)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7,
                                out_layer8, out_layer9, out_layer10, out_layer11, out_layer12, out_layer13],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB3_BotNeck14_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB3_BotNeck14_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB3_BotNeck14_Act')(bottleneck)
    out_layer14 = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB3_Layer14_Conv')(bottleneck)
    out_layer14 = BatchNormalization(name='DB3_Layer14_BN')(out_layer14)
    out_layer14 = Activation('relu', name='DB3_Layer14_Act')(out_layer14)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7,
                                out_layer8, out_layer9, out_layer10, out_layer11, out_layer12, out_layer13, out_layer14],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB3_BotNeck15_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB3_BotNeck15_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB3_BotNeck15_Act')(bottleneck)
    out_layer15 = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB3_Layer15_Conv')(bottleneck)
    out_layer15 = BatchNormalization(name='DB3_Layer15_BN')(out_layer15)
    out_layer15 = Activation('relu', name='DB3_Layer15_Act')(out_layer15)

    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7,
                                out_layer8, out_layer9, out_layer10, out_layer11, out_layer12, out_layer13, out_layer14, out_layer15],axis=-1)
    bottleneck  = Conv3D(bottleneck_factor * growth_rate, kernel_size=1, strides=1, padding='valid', name='DB3_BotNeck16_Conv')(input_cat)
    bottleneck  = BatchNormalization(name='DB3_BotNeck16_BN')(bottleneck)
    bottleneck  = Activation('relu', name='DB3_BotNeck16_Act')(bottleneck)
    out_layer16 = Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', name='DB3_Layer16_Conv')(bottleneck)
    out_layer16 = BatchNormalization(name='DB3_Layer16_BN')(out_layer16)
    out_layer16 = Activation('relu', name='DB3_Layer16_Act')(out_layer16)
    out_layer16 = Dropout(rate=0.2)(out_layer16)

    # Upsampling from (30,18,30) -> (60,36,60)
    ### Which output is being passed from previous DenseBlock?
    # 1. The concatenation of all the previous feature maps or
    # 2. Only the output feature map of the last layer within the DenseBlock
    input_cat   = concatenate([out_trans, out_layer1, out_layer2, out_layer3, out_layer4, out_layer5, out_layer6, out_layer7,
                                out_layer8, out_layer9, out_layer10, out_layer11, out_layer12, out_layer13, out_layer14, out_layer15, out_layer16],axis=-1)
    
    input_cat   = BatchNormalization()(input_cat)
    input_cat   = Activation('relu')(input_cat)
    up_1        = Conv3DTranspose(64, kernel_size=(2, 2, 2), strides=(2,2,2))(input_cat)

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
    fin = DenseNet(input_tsdf)
    model = Model(inputs=input_tsdf, outputs=fin)
    return model

def get_network_by_name2(name):
    if name == 'DENSENET':
        return get_densenet(), 'depth'

def get_net_name_from_w2(weights):
    networks = ['DENSENET']

    for net in networks:
        if ((net+'_LR') == (weights[0:len(net)+3])) or ((net+'fine_LR') == (weights[0:len(net)+7])):
            return net

    print("Invalid weight file: ", weights)
    exit(-1)
