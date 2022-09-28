from logging import raiseExceptions
import tensorflow as tf
import weightFreeze
from glob import glob

def set_gpu_config():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
def define_model(model_config, gpuNum):
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

    if model_config['mType'] == 'singleLayer':
        from getSingleLayerNet import getSingleLayerNet
        initializer = tf.keras.initializers.TruncatedNormal(mean=0.0,stddev=0.25)
        with tf.device(device_name):
            model = getSingleLayerNet(model_config['num_dim'], initializer, model_config['hiddenSize'])
    elif model_config['mType'] == 'AlexNet':
        from AlexNet import AlexNet

        with tf.device(device_name):
            if model_config['pre_trained']:
                channels = 3
                inputSize = 227
                num_classes = 10
                model = AlexNet((inputSize,inputSize,channels), num_classes, True) # will load weights
            else:
                channels = model_config['channels']
                image_width = model_config['image_width']
                image_height = model_config['image_height']
                num_classes = model_config['num_classes']
                model = AlexNet((image_width,image_height,channels), num_classes, False) # will not load weights          
    else:
        raise NameError('Wrong model type not specified yet!')

    return model

def get_train_param(expr_name):
    if expr_name == 'singleLayer':
        return 0.03, tf.keras.losses.BinaryCrossentropy(from_logits=False)
    elif expr_name == 'AlexNet_pretrained':
        return 0.001, tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    elif expr_name == 'AlexNet_vanilla':
        return 0.001, tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    else:
        raise NameError('Wrong expr name specified!')


def train_model(model, num_epoch, expr_name, ds_train, ds_test, callback_list, gpuNum, **kwargs):
    if 'start_epoch' in kwargs:
        start_epoch = kwargs['start_epoch']
    else:
        start_epoch = 0 # still 0-start
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
        lr, loss_fn = get_train_param(expr_name)   
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss=loss_fn,
            metrics=['accuracy'],
        )

        model.fit(
            ds_train,
            epochs=num_epoch,
            verbose = False,
            validation_data=ds_test,
            callbacks=callback_list,
            initial_epoch=start_epoch
        )
    return model

def eachIter_dualCond(log_dir, chkpt_dir, model_config, ds_train, ds_test, num_epoch, resetType, doLog, gpuNum, doEarlyStopping, **kwargs):
    if 'ckpt_freq' in kwargs:
        ckpt_freq = kwargs['ckpt_freq']
    else:
        ckpt_freq = 1
        
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
    model_init_freeze = define_model(model_config, gpuNum)
    tensorboard_callback_freeze = tf.keras.callbacks.TensorBoard(log_dir=log_dir.format(typeStr = 'freeze'), write_graph=False, update_freq='epoch', histogram_freq = int(doLog)) # if log the weights here, it will overwrite
    cp_callback_freeze = tf.keras.callbacks.ModelCheckpoint(
        chkpt_dir.format(typeStr = 'freeze') + "/cp-{epoch:04d}.ckpt",
        monitor = 'val_accuracy',
        save_best_only=False,
        verboe=0,
        save_weights_only=True,
        save_freq = 'epoch',
        period = ckpt_freq
    )
    if 'doRandInit' in kwargs:
        wf_callback = weightFreeze.weightFreeze(ds_train_freeze, ds_test, log_dir.format(typeStr = 'freeze'), resetType, True, doLog, doLog, doRandInit = kwargs['doRandInit'])
    if doLog: # if log batch info, won't use tensorboard callback at all as it will always overwrite weight here no matter what. 
        callback_list_freeze = [wf_callback, cp_callback_freeze]
    else:
        callback_list_freeze = [wf_callback, tensorboard_callback_freeze, cp_callback_freeze]

    # prepare for liquid
    model_init_liquid = tf.keras.models.clone_model(model_init_freeze)
    model_init_liquid.set_weights(model_init_freeze.get_weights())
    tensorboard_callback_liquid = tf.keras.callbacks.TensorBoard(log_dir=log_dir.format(typeStr = 'liquid'), write_graph=False, update_freq='epoch', histogram_freq = int(doLog)) # writing image does take a long time....
    cp_callback_liquid = tf.keras.callbacks.ModelCheckpoint(
        chkpt_dir.format(typeStr = 'liquid') + "/cp-{epoch:04d}.ckpt",
        monitor = 'val_accuracy',
        save_best_only=False,
        verboe=0,
        save_weights_only=True,
        save_freq = 'epoch',
        period = ckpt_freq
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

    if model_config['mType'] == 'singleLayer':
        expr_name = 'singleLayer'
    elif model_config['mType'] == 'AlexNet':
        if model_config['pre_trained']:
            expr_name = 'AlexNet_pretrained'
        else:
            expr_name = 'AlexNet_vanilla'
    else:
        raise NameError('No match between model config and expr name!')

    ## This part adds for picking up from previous training. 
    start_epoch = 0 # epoch is 0-start despite how the checkpoints appear!!
    if 'start_epoch' in kwargs:
        start_epoch = kwargs['start_epoch']
        if not start_epoch == 0: # asks for loading the checkpoints!
            for typeStr in ['freeze', 'liquid']:
                ckpt_path = chkpt_dir.format(typeStr = typeStr) + "/cp-{epoch:04d}.ckpt".format(epoch = start_epoch)
                ckpts = glob(ckpt_path + '.*')
                if len(ckpts) > 0:
                    if typeStr == 'freeze':
                        model_init_freeze.load_weights(ckpt_path)
                    else:
                        model_init_liquid.load_weights(ckpt_path)
                else:
                    raiseExceptions('Requested checkpoint %d for %s doesnt exist!' % (start_epoch, typeStr))
                            
    model_freeze = train_model(model_init_freeze, num_epoch, expr_name, ds_train_freeze, ds_test, callback_list_freeze, gpuNum, start_epoch=start_epoch)
    model_liquid = train_model(model_init_liquid, num_epoch, expr_name, ds_train_liquid, ds_test, callback_list_liquid, gpuNum, start_epoch=start_epoch)
    return model_freeze, model_liquid