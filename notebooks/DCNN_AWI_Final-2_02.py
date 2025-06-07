
import numpy as np
import os
import sys
import struct
import tqdm
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from tensorflow.keras import mixed_precision
import scipy.interpolate
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model


# # normalize

def normal_noise(lst):
  m=max(lst)
  for i in range(len(lst)):
    lst[i]/=m

def power_cal(lst,rc):
  x=list(range(128))
  y_interp = scipy.interpolate.interp1d(x, lst)
  return y_interp(rc)


# # Read all data#

range_step=0.468
range_m=[]
for i in range(128):
    range_m.append(i*range_step)


# # Read data

import numpy as np
import os
from tqdm import tqdm


#path="/albedo/work/user/aldehg001/Data/TEST/New_Shuffle_data/DNN_INPUT_CY_LRM_C_SIM_FLOOR_new_shuffle/"
path="/albedo/work/user/aldehg001/Data/TEST/New_Shuffle_data/DNN_INPUT_CY_LRM_C_SIM_1000_25_FLOOR/"
#path="/albedo/work/user/aldehg001/Data/TEST/New_Shuffle_data/DNN_INPUT_S3A_PLRM_SIM_1000_25_110_FLOOR/"

ATT_lst=[]
Sim_NO_noise_lst=[]
SIM_WF_NOISE_lst=[]
OCOG_lst=[]
OCOG_NOISE_lst=[]
REF_lst=[]
lst_file=os.listdir(path)
lst_file.sort()
lst_number=[]
#TFMRA_NOISE_lst=[]

for item_name in tqdm (lst_file):
  if("ATT" in item_name ):
    idx_file=item_name.split("_")[-1].split(".")[0]
    lst_number.append(idx_file)
    
    #------------------------------ATT read-----------------------------------#
    
    file=open(path+"ATT_ARR_"+str(idx_file)+".bin","rb")
    array_att_arr=np.fromfile(file,np.double)
    n_att=array_att_arr.size
    ATT_lst.append(array_att_arr)
    file.close()

    #--------------------------SIM_WF_NO_NOISE--------------------------------#
    
    file=open(path+"SIM_WF_NO_NOISE_"+str(idx_file)+".bin","rb")
    array_temp=np.fromfile(file,np.float32)
    n_samples=int(array_temp.size/n_att)
    array_sim_wf_no_noise=np.reshape(array_temp,[n_att,n_samples])
    Sim_NO_noise_lst.append(array_sim_wf_no_noise)
    file.close()

    #---------------------------SIM_WF_NOISE---------------------------------#
    
    file=open(path+"SIM_WF_NOISE_"+str(idx_file)+".bin","rb")
    array_temp=np.fromfile(file,np.float32)
    n_noise=int(array_temp.size/(n_att*n_samples))
    array_sim_wf_noise=np.reshape(array_temp,[n_att,n_noise,n_samples])
    for i in range(n_att):
      for j in range(n_noise):
        normal_noise(array_sim_wf_noise[i][j])
    SIM_WF_NOISE_lst.append(array_sim_wf_noise)
    file.close()


    #-----------------------------------OCOG---------------------------------#
    
    file=open(path+"OCOG_ARR_"+str(idx_file)+".bin","rb")
    array_ocog_arr=np.fromfile(file,np.double)
    n_ocog=int(array_ocog_arr.size)
    OCOG_lst.append(array_ocog_arr)
    file.close()
    
    #--------------------------------REF_ARR---------------------------------#
    
    file=open(path+"REF_ARR_"+str(idx_file)+".bin","rb")
    array_ref_arr=np.fromfile(file,np.double)
    n_ref=int(array_ref_arr.size)
    file.close()
    REF_lst.append(array_ref_arr)

    #-----------------------------OCOG Noisy---------------------------------#
    
    file=open(path+"OCOG_ARR_NOISE_"+str(idx_file)+".bin","rb")
    array_temp=np.fromfile(file,np.double)
    array_ocog_noise_arr=np.reshape(array_temp,[n_att,n_noise])
    array_ocog_noise_arr=np.delete(array_ocog_noise_arr,list(range(50,array_ocog_noise_arr.shape[1])),1)
    OCOG_NOISE_lst.append(array_ocog_noise_arr)
    file.close()

    #----------------------------TFMRA--------------------------------------#
    
#    file=open(path+"TFMRA_ARR_NOISE_"+str(idx_file)+".bin","rb")
#    array_temp=np.fromfile(file,np.double)
#    array_tfmra_noise_arr=np.reshape(array_temp,[n_att,n_noise])
#    array_tfmra_noise_arr=np.delete(array_tfmra_noise_arr,list(range(50,array_tfmra_noise_arr.shape[1])),1)
#    TFMRA_NOISE_lst.append(array_tfmra_noise_arr)
#    file.close()
    
    
    
    #file=open(path+"OCOG_ARR_NOISE_"+str(idx_file)+".bin","rb")
    #array_temp=np.fromfile(file,np.double)
    #array_ocog_noise_arr=np.reshape(array_temp,[n_att,n_att])
    #array_ocog_noise_arr=np.delete(array_ocog_noise_arr,list(range(50,array_ocog_noise_arr.shape[1])),1)
    #OCOG_NOISE_lst.append(array_ocog_noise_arr)
    #file.close()

    #------------------------------------------------------------------------#

print("   files")
print("_"*100)
print("\u2713  Read ATT_ARR file")
print("\u2713  Read SIM_WF_NO_NOISE file")
print("\u2713  Read SIM_WF_NOISE file")
print("\u2713  Read OCOG_ARR file")
print("\u2713  Read OCOG_ARR_NOISE file")
print("\u2713  Read REF_ARR file")
print("_"*100)




ATT_lst=np.array(ATT_lst)
Sim_NO_noise_lst=np.array(Sim_NO_noise_lst)
SIM_WF_NOISE=np.array(SIM_WF_NOISE_lst)
OCOG_lst=np.array(OCOG_lst)
OCOG_NOISE_lst=np.array(OCOG_NOISE_lst)
REF_lst=np.array(REF_lst)


print("       Data.shape ")
print("_"*100)
print("_"*100)

print("  \u2713    ATT_lst: {0}".format(ATT_lst.shape))
print("  \u2713    Sim_NO_noise_lst: {0}".format(Sim_NO_noise_lst.shape))
print("  \u2713    SIM_WF_NOISE: {0}".format(SIM_WF_NOISE.shape))
print("  \u2713    OCOG_lst: {0}".format(OCOG_lst.shape))
print("  \u2713    OCOG_NOISE_lst: {0}".format(OCOG_NOISE_lst.shape))
print("  \u2713    REF_lst: {0}".format(REF_lst.shape))

print("_"*100)
print("_"*100)


print(" location")
print("_"*100)

lst_number=np.array(lst_number)
lst_number



#----------


ATT_lst=np.array(ATT_lst)
Sim_NO_noise_lst=np.array(Sim_NO_noise_lst)
SIM_WF_NOISE=np.array(SIM_WF_NOISE_lst)
OCOG_lst=np.array(OCOG_lst)
OCOG_NOISE_lst=np.array(OCOG_NOISE_lst)
REF_lst=np.array(REF_lst)


print("       Data.shape ")
print("_"*100)
print("_"*100)

print("  \u2713    ATT_lst: {0}".format(ATT_lst.shape))
print("  \u2713    Sim_NO_noise_lst: {0}".format(Sim_NO_noise_lst.shape))
print("  \u2713    SIM_WF_NOISE: {0}".format(SIM_WF_NOISE.shape))
print("  \u2713    OCOG_lst: {0}".format(OCOG_lst.shape))
print("  \u2713    OCOG_NOISE_lst: {0}".format(OCOG_NOISE_lst.shape))
print("  \u2713    REF_lst: {0}".format(REF_lst.shape))

print("_"*100)
print("_"*100)



# # Get all data

# extract 50 noise from each wave 
data_wave=[]
lbl_OCOG=[]
lbl_REF=[]
for k in range(ATT_lst.shape[0]):
  array_sim_wf_no_noise=Sim_NO_noise_lst[k]
  print(k)
  array_ocog_arr=OCOG_lst[k]
  array_REF_arr=REF_lst[k]
  array_sim_wf_noise=SIM_WF_NOISE[k]
  for i in range(array_sim_wf_no_noise.shape[0]):
   # data_wave.append(array_sim_wf_no_noise[i])
   # lbl_OCOG.append(array_ocog_arr[i])
   # lbl_REF.append(array_REF_arr[i])
# get all noise from wave
    for item in array_sim_wf_noise[i]:
      data_wave.append(item) 
      lbl_OCOG.append(array_ocog_arr[i])
      lbl_REF.append(array_REF_arr[i])


data_wave=np.array(data_wave)
lbl_REF=np.array(lbl_REF)


print("_"*100)
print(" \u2713  Final Data shape: {0}".format(data_wave.shape))
print(" \u2713  Final lbl_REF shape: {0}".format(lbl_REF.shape))
print("_"*100)

lbl= lbl_REF


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_wave, lbl,
                                                    test_size=0.20,
                                                    shuffle = False, stratify = None,
                                                    random_state=0)


print("_"*100)
print("   Final  x_train shape: {0}".format(x_train.shape))
print("   Final  x_test  shape: {0}".format(x_test.shape))
print("   Final  x_train shape: {0}".format(y_train.shape))
print("   Final  x_test  shape: {0}".format(y_test.shape))
print("_"*100)


# # CNN

input_size=data_wave.shape[1],1
lr_value=2e-4
batch_size_value=128
epochs_value=30


def create_cnn_model():
  input_layer=keras.layers.Input(shape=input_size,name="input_layer")

  # --------------------------Block 1----------------------------
    
  x = keras.layers.Conv1D(64, (3), 
                    kernel_initializer='he_normal',
                    strides=1,
                    padding='same',
                    name='conv1.1')(input_layer)
  x=keras.layers.Activation("relu")(x)
  x=keras.layers.BatchNormalization()(x)

  x = keras.layers.Conv1D(64, (3), 
                    kernel_initializer='he_normal',
                    strides=1,
                    padding='same',
                    name='conv1.2')(x)
  x=keras.layers.Activation("relu")(x)
  x=keras.layers.BatchNormalization()(x)

  x = keras.layers.Conv1D(64, (3), 
                    kernel_initializer='he_normal',
                    strides=1,
                    padding='same',
                    name='conv1.3')(x)
  x=keras.layers.Activation("relu")(x)
  x=keras.layers.BatchNormalization()(x)

  x = keras.layers.MaxPooling1D((2), strides=(2), name='block1_pool')(x)
  x = keras.layers.Dropout(0.25)(x)

  # --------------------------Block 2----------------------------

  x = keras.layers.Conv1D(96, (3), 
                    kernel_initializer='he_normal',
                    strides=1,
                    padding='same',
                    name='conv2.1')(x)
  x = keras.layers.Activation("relu")(x)
  x=keras.layers.BatchNormalization()(x)

  x = keras.layers.Conv1D(96, (3), 
                    kernel_initializer='he_normal',
                    strides=1,
                    padding='same',
                    name='conv2.2')(x)
  x = keras.layers.Activation("relu")(x)
  x=keras.layers.BatchNormalization()(x)

  x = keras.layers.MaxPooling1D((2), strides=(2), name='block2_pool')(x)
  x = keras.layers.Dropout(0.25)(x)


  # ---------------------------Block 3-----------------------------
    
  x = keras.layers.Conv1D(128, (3), 
                    kernel_initializer='he_normal',
                    strides=1,
                    padding='same',
                    name='conv3.1')(x)
  x=keras.layers.Activation("relu")(x)
  x=keras.layers.BatchNormalization()(x)

  x = keras.layers.Conv1D(128, (3),
                    kernel_initializer='he_normal',
                    strides=1,
                    padding='same',
                    name='conv3.2')(x)
  x=keras.layers.Activation("relu")(x)
  x=keras.layers.BatchNormalization()(x)

  x = keras.layers.MaxPooling1D((2), strides=(2), name='block3_pool')(x)
  x = keras.layers.Dropout(0.25)(x)


  # ----------------------------Block 4------------------------------

  x = keras.layers.Conv1D(160, (3),
                    kernel_initializer='he_normal',
                    strides=1,
                    padding='same',
                    name='conv4.1')(x)
  x=keras.layers.Activation("relu")(x)
  x=keras.layers.BatchNormalization()(x)

  x = keras.layers.Conv1D(160, (3),
                    kernel_initializer='he_normal',
                    strides=1,
                    padding='same',
                    name='conv4.2')(x)
  x=keras.layers.Activation("relu")(x)
  x=keras.layers.BatchNormalization()(x)

  x = keras.layers.MaxPooling1D((2), strides=(2), name='block4_pool')(x)
  x = keras.layers.Dropout(0.25)(x)

  # -----------------------------Block 5-------------------------------
    
  x = keras.layers.Conv1D(192, (3), 
                    kernel_initializer='he_normal',
                    strides=1,
                    padding='same',
                    name='conv5.1')(x)
  x=keras.layers.Activation("relu")(x)
  x=keras.layers.BatchNormalization()(x)

  x = keras.layers.Conv1D(192, (3),
                    kernel_initializer='he_normal',
                    strides=1,
                    padding='same',
                    name='conv5.2')(x)
  x=keras.layers.Activation("relu")(x)
  x=keras.layers.BatchNormalization()(x)

  x = keras.layers.MaxPooling1D((2), strides=(2), name='block5_pool')(x)
  x = keras.layers.Dropout(0.25)(x)

 # ------------------------------Block 6-------------------------------
  x = keras.layers.Conv1D(224, (3),
                    kernel_initializer='he_normal',
                    strides=1,
                    padding='same',
                    kernel_regularizer='l2',
                    name='conv6.1')(x)
  x=keras.layers.Activation("relu")(x)
  x=keras.layers.BatchNormalization()(x)
  x = keras.layers.MaxPooling1D((2), strides=(2), name='block6_pool')(x)
  x = keras.layers.Dropout(0.25)(x)
  flatten = keras.layers.Flatten()(x)
  fc1 = keras.layers.Dense(128, activation='relu')(flatten)

  #____________________________________________________________
  output = keras.layers.Dense(1)(fc1)
  CNN_model = keras.Model(inputs=input_layer, outputs=output)
  optimizer = keras.optimizers.Adam(learning_rate=lr_value)
  CNN_model.compile(loss="mse",
                     optimizer=optimizer)
  return CNN_model


CNN_model=create_cnn_model()
CNN_model.summary()


# # Early Stopping

class stop_early(keras.callbacks.Callback):
  def _init_(self):
    super(stop_early,self)._init_()
  def on_epoch_end(self,epoch,logs=None):
    if(epoch>=30  and logs["val_loss"]<0.07):
      self.model.stop_training = True
custom_early_stopping=stop_early()


hist_CNN=CNN_model.fit(x_train,y_train,shuffle=True,
                                     batch_size=batch_size_value,
                                     epochs=epochs_value,
                                     callbacks=[custom_early_stopping],
                                     validation_data=(x_test,y_test),
                                     )

np.save(path+"history_DCNN_pool2_2_01.npy", hist_CNN.history)

CNN_model.save(path+'DCNN_pool_2_2_01.h5')

# # END
