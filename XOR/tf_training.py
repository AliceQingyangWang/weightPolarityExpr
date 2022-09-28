import tensorflow as tf
from getSingleLayerNet import getSingleLayerNet
import weightFreeze

def set_gpu_config():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
def define_model(inputSize, gpuNum, hiddenSize=64):
    initializer = tf.keras.initializers.TruncatedNormal(mean=0.0,stddev=0.25)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        if isinstance(gpuNum, int):
            device_name = '/GPU:{gpuNum:d}'.format(gpuNum = gpuNum)
            print('Using selected GPU {gpuNum}!!'.format(gpuNum = gpuNum))
        else:
            gpu_name = gpus[0].name
            device_name = gpu_name[0] + gpu_name[17:]
    else:
        device_name = '/CPU:0'
    with tf.device(device_name):
        model = getSingleLayerNet(inputSize, initializer, hiddenSize)

    return model

def train_model(model, num_epoch, num_classes, ds_train, ds_test, callback_list, gpuNum):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:        
        if isinstance(gpuNum, int):
            device_name = '/GPU:{gpuNum:d}'.format(gpuNum = gpuNum)
            print('Using selected GPU {gpuNum}!!'.format(gpuNum = gpuNum))
        else:
            gpu_name = gpus[0].name
            device_name = gpu_name[0] + gpu_name[17:]
    else:
        device_name = '/CPU:0'

    with tf.device(device_name):#'/GPU:0'        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.03),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )

        model.fit(
            ds_train,
            epochs=num_epoch,
            verbose = False,
            validation_data=ds_test,
            callbacks=callback_list
        )
    return model

# def eachIter(log_dir, chkpt_dir, ds_train, ds_test, inputSize, num_epoch, resetType, doLog, gpuNum, hiddenSize=64, **kwargs):
#     tf.debugging.set_log_device_placement(True)
#     if isinstance(ds_train, list):
#         if len(ds_train) == 2:
#             ds_train_freeze = ds_train[0]
#         else:
#             raise ValueError
#     else:
#         ds_train_freeze = ds_train

#     # prepare for freeze
#     model_init_freeze = define_model(inputSize, gpuNum, hiddenSize)
#     tensorboard_callback_freeze = tf.keras.callbacks.TensorBoard(log_dir=log_dir.format(typeStr = 'freeze'), write_graph=False, update_freq='epoch', histogram_freq = int(not doLog)) # if log the weights here, it will overwrite
#     cp_callback_freeze = tf.keras.callbacks.ModelCheckpoint(
#         chkpt_dir.format(typeStr = 'freeze') + "/cp-{epoch:04d}.ckpt",
#         monitor = 'val_accuracy',
#         save_best_only=False,
#         verboe=0,
#         save_weights_only=True,
#         save_freq = 'epoch'
#     )
#     if 'doRandInit' in kwargs:
#         wf_callback = weightFreeze.weightFreeze(ds_train_freeze, ds_test, log_dir.format(typeStr = 'freeze'), resetType, True, doLog, doLog, doRandInit = kwargs['doRandInit'])
#     if doLog: # if log batch info, won't use tensorboard callback at all as it will always overwrite weight here no matter what. 
#         callback_list_freeze = [wf_callback, cp_callback_freeze]
#     else:
#         callback_list_freeze = [wf_callback, tensorboard_callback_freeze, cp_callback_freeze]

#     model_freeze = train_model(model_init_freeze, num_epoch, 2, ds_train_freeze, ds_test, callback_list_freeze, gpuNum)
#     return model_freeze

def eachIter_dualCond(log_dir, chkpt_dir, ds_train, ds_test, inputSize, num_epoch, resetType, doLog, gpuNum, doEarlyStopping, hiddenSize=64, **kwargs):
    tf.debugging.set_log_device_placement(True)
    if isinstance(ds_train, list):
        if len(ds_train) == 2:
            ds_train_freeze = ds_train[0]
            ds_train_liquid = ds_train[1]
        else:
            raise ValueError
    else:
        ds_train_freeze = ds_train
        ds_train_liquid = ds_train

    # prepare for freeze
    model_init_freeze = define_model(inputSize, gpuNum, hiddenSize)
    tensorboard_callback_freeze = tf.keras.callbacks.TensorBoard(log_dir=log_dir.format(typeStr = 'freeze'), write_graph=False, update_freq='epoch', histogram_freq = int(not doLog)) # if log the weights here, it will overwrite
    cp_callback_freeze = tf.keras.callbacks.ModelCheckpoint(
        chkpt_dir.format(typeStr = 'freeze') + "/cp-{epoch:04d}.ckpt",
        monitor = 'val_accuracy',
        save_best_only=False,
        verboe=0,
        save_weights_only=True,
        save_freq = 'epoch'
    )
    if 'doRandInit' in kwargs:
        wf_callback = weightFreeze.weightFreeze(ds_train_freeze, ds_test, log_dir.format(typeStr = 'freeze'), resetType, True, doLog, doLog, doRandInit = kwargs['doRandInit'])
    if doLog: # if log batch info, won't use tensorboard callback at all as it will always overwrite weight here no matter what. 
        callback_list_freeze = [wf_callback, cp_callback_freeze]
    else:
        callback_list_freeze = [wf_callback, tensorboard_callback_freeze, cp_callback_freeze]

    # prepare for liquid
    model_init_liquid = tf.keras.models.clone_model(model_init_freeze)
    tensorboard_callback_liquid = tf.keras.callbacks.TensorBoard(log_dir=log_dir.format(typeStr = 'liquid'), write_graph=False, update_freq='epoch', histogram_freq = int(not doLog)) # writing image does take a long time....
    cp_callback_liquid = tf.keras.callbacks.ModelCheckpoint(
        chkpt_dir.format(typeStr = 'liquid') + "/cp-{epoch:04d}.ckpt",
        monitor = 'val_accuracy',
        save_best_only=False,
        verboe=0,
        save_weights_only=True,
        save_freq = 'epoch'
    )
    if 'doRandInit' in kwargs:
        wf_callback = weightFreeze.weightFreeze(ds_train_liquid, ds_test, log_dir.format(typeStr = 'liquid'), resetType, False, doLog, doLog, doRandInit = kwargs['doRandInit'])
    if doLog:
        callback_list_liquid = [wf_callback, cp_callback_liquid]
    else:    
        callback_list_liquid = [wf_callback, tensorboard_callback_liquid, cp_callback_liquid]

    if doEarlyStopping:
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                                    monitor='val_loss',
                                    min_delta=0,
                                    patience=10,
                                    verbose=0,
                                    mode='auto',
                                    baseline=None,
                                    restore_best_weights=False
                                )
        callback_list_freeze.append(early_stopping_callback)
        callback_list_liquid.append(early_stopping_callback)
                            
    model_freeze = train_model(model_init_freeze, num_epoch, 2, ds_train_freeze, ds_test, callback_list_freeze, gpuNum)
    model_liquid = train_model(model_init_liquid, num_epoch, 2, ds_train_liquid, ds_test, callback_list_liquid, gpuNum)
    return model_freeze, model_liquid