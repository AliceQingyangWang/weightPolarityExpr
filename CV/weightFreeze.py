import keras
import numpy as np
import tensorflow as tf
import os
from helperfun import plot_to_image, plot_weights

class weightFreeze(keras.callbacks.Callback):
    def __init__(self, train_data, val_data, log_dir, resetType, doWeightFreeze, logFreeW, logResetW, **kwargs):
        super().__init__()
        self.train_data = train_data
        self.validation_data = val_data
        self.polarityConfig = {}
        self.logFreeW = logFreeW & doWeightFreeze
        self.logResetW = logResetW
        self.doWeightFreeze = doWeightFreeze
        self.resetType = resetType
        self.epsilon = 0.001
        self.doRandInit = False
        if 'doRandInit' in kwargs:
            if kwargs['doRandInit']:
                self.doRandInit = True
        # print(logFreeW)
        # so clearly tensorboard ppl broke their own documented api and it can no longer log batch level info
        # will do it mannually here....
        self._writers = {}
        # if self.logResetW:
        self._writers_dir = {}
        self._writers_dir['train'] = os.path.join(log_dir, 'train')
        self._writers_dir['validation'] = os.path.join(log_dir, 'validation')
        self._writers['train'] = tf.summary.create_file_writer(self._writers_dir['train'])
        self._writers['validation'] = tf.summary.create_file_writer(self._writers_dir['validation'])

        if self.logFreeW:
            self._writers_dir['train_freeW'] = os.path.join(log_dir, 'train_freeW')
            self._writers_dir['validation_freeW'] = os.path.join(log_dir, 'validation_freeW')
            self._writers['train_freeW'] = tf.summary.create_file_writer(self._writers_dir['train_freeW'])
            self._writers['validation_freeW'] = tf.summary.create_file_writer(self._writers_dir['validation_freeW'])

    def _train_writer(self, logSelector):
        writerName = 'train' + logSelector
        if writerName not in self._writers:
            self._writers[writerName] = tf.summary.create_file_writer(
                self._writers_dir[writerName]) 
        return self._writers[writerName]

    def _val_writer(self, logSelector):
        writerName = 'validation' + logSelector
        if writerName not in self._writers:
            self._writers[writerName] = tf.summary.create_file_writer(self._writers_dir[writerName]) 
        return self._writers[writerName]

    def _log_epoch_metrics(self, epoch, logs, logSelector):
        """Writes epoch metrics out as scalar summaries.
        Args:
            epoch: Int. The global step to use for TensorBoard.
            logs: Dict. Keys are scalar summary names, values are scalars.
        """
        ### copied from tensorboard
        if not logs:
            return

        train_logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        val_logs = {k: v for k, v in logs.items() if k.startswith('val_')}

        with tf.summary.record_if(True):
            if train_logs:
                with (self._train_writer(logSelector)).as_default():
                    for name, value in train_logs.items():
                        tf.summary.scalar('epoch_' + name, data=value, step=epoch)
            if val_logs:
                with (self._val_writer(logSelector)).as_default():
                    for name, value in val_logs.items():
                        name = name[4:]  # Remove 'val_' prefix.
                        tf.summary.scalar('epoch_' + name, data=value, step=epoch)   
                        
    def _log_batch_metrics(self, batch, logs, logSelector):
        """Writes epoch metrics out as scalar summaries.
        Args:
            epoch: Int. The global step to use for TensorBoard.
            logs: Dict. Keys are scalar summary names, values are scalars.
        """
        ### copied from tensorboard
        if not logs:
            return

        train_logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        val_logs = {k: v for k, v in logs.items() if k.startswith('val_')}

        with tf.summary.record_if(True):
            if train_logs:
                with (self._train_writer(logSelector)).as_default():
                    for name, value in train_logs.items():
                        tf.summary.scalar('batch_' + name, data=value, step=batch)
            if val_logs:
                with (self._val_writer(logSelector)).as_default():
                    for name, value in val_logs.items():
                        name = name[4:]  # Remove 'val_' prefix.
                        tf.summary.scalar('batch_' + name, data=value, step=batch)   

    def on_train_begin(self, logs=None):
        # log accuracy before reset
        train_logs = self.model.evaluate(self.train_data)
        val_logs = self.model.evaluate(self.validation_data)
        self._log_epoch_metrics(-2, {'accuracy':train_logs[1], 'val_accuracy':val_logs[1]}, '')

        # initial resetting will only happen if you are doing polarity freezing - because this is the only place you have to make sure you start with the correct one. the other scenario can always find its own place
        if self.doWeightFreeze:

            if self.model.name == 'alex_net': 
                config_model = self.model.clone()
                config_model.set_weights(self.model.get_weights()) # this ensures the last layer has the correct weights!
                if not self.doRandInit:
                    config_model.load_weights_pickle()
                            
            for layer in self.model.layers:
                if isinstance(layer, (keras.layers.Dense, keras.layers.Conv2D)):
                    [layer_weights, layer_bias] = layer.get_weights()

                    # print(layer.name) 
                    # weight already have their polarity. can just use that as a random initialization.
                    # just flip the necessary few to get the configuration right, this shouldn't affect the gaussian assumption too much

                    if self.model.name == 'singleLayer':
                        thisLConfig = np.sign(layer_weights)

                        if not self.doRandInit:
                            # the rule: if dense_1 has opposite signs, make the connection positive, otherwise, negative
                            if layer.name == 'dense_1':
                                preLConfig = thisLConfig
                            elif layer.name == 'dense_2':
                                thisLConfig = np.reshape(np.sign(np.multiply(preLConfig[0,:], preLConfig[1, :])), [-1,1])
                                # print(preLConfig.shape)
                                # print(thisLConfig.shape)
                                # print(layer_weights.shape)
                                # print(preLConfig[0,0])
                                # print(preLConfig[1,0])
                                # print(thisLConfig[0])
                            else:
                                raise NameError('Layer name is not reognized. problem with setting network layer name!')
                    elif self.model.name == 'alex_net':
                        thisLConfig = np.sign(config_model.get_layer(layer.name).get_weights()[0]) 
                    else:
                        raise NameError('Model name not recognized!')

                    self.polarityConfig[layer.name] = thisLConfig
                    checker = np.multiply(layer_weights, thisLConfig)
                    layer_weights = self.check_polarity(layer_weights, checker)

                    layer.set_weights([layer_weights, layer_bias])

            del config_model
        # log accuracy after reset, before training
        train_logs = self.model.evaluate(self.train_data)
        val_logs = self.model.evaluate(self.validation_data)
        self._log_epoch_metrics(-1, {'accuracy':train_logs[1], 'val_accuracy':val_logs[1]}, '')
        log_path = self._writers_dir['train'].split('/')
        log_path[-4] = 'checkpoints'
        del log_path[-1]
        check_path = os.sep.join(log_path)
        self.model.save_weights(os.path.join(check_path, 'cp-0000.ckpt'))

    def check_polarity(self, layer_weights, checker):
        if self.resetType == 'zero':
            layer_weights[checker < 0] = 0 #this should set the necessary weight sto zero
        elif self.resetType == 'flip':
            layer_weights[checker < 0] = -layer_weights[checker < 0] #this should flip the necessary weights
        elif self.resetType == 'posRand':
            rand_sub = np.random.uniform(low=0.0, high=self.epsilon, size=layer_weights.shape)
            # first flip weight matrix sign
            layer_weights[checker < 0] = -layer_weights[checker < 0]
            rand_sub[layer_weights < 0] = -rand_sub[layer_weights < 0] # make sure the epsilon shares the correct sign as the weihgt matrix
            layer_weights[checker < 0] = rand_sub[checker < 0] #set to random correct-sign number
        elif self.resetType == 'posCon':
            weight_sign = np.sign(layer_weights)
            layer_weights[np.logical_and(checker < 0, weight_sign < 0)] = self.epsilon #set to random positive number, this is so stupid..... you are changing these guys twice..... Now should be correct.....
            layer_weights[np.logical_and(checker < 0, weight_sign > 0)] = -self.epsilon
        else:
            raise ValueError     
        return layer_weights

    def normalize_tensor(self, tensor):
        return tf.divide(
            tf.subtract(
                tensor,
                tf.reduce_min(tensor),
            ),
            tf.subtract(
                tf.reduce_max(tensor),
                tf.reduce_min(tensor)
            )
        )

    def on_train_batch_end(self, batch, logs):
       
        if self.logFreeW:# only log if actually freezing the weights, otherwise don't log at all 
            val_logs = self.model.evaluate(self.validation_data)
            val_logs = {'val_loss': val_logs[0], 'val_accuracy': val_logs[1]}
            logs.update(val_logs)
            self._log_batch_metrics(batch, logs, '_freeW') 

        for layer in self.model.layers:
            layer.trainable = False
            if isinstance(layer, (keras.layers.Dense, keras.layers.Conv2D)):
                [layer_weights, layer_bias] = layer.get_weights()

                if self.doWeightFreeze:
                    if self.logFreeW:
                        with self._train_writer('_freeW').as_default():
                            tf.summary.histogram(layer.name+'/kernel', layer_weights, step=batch)
                            # wTemp = tf.cast(layer_weights, dtype = tf.float32)      
                            # wTemp = self.normalize_tensor(wTemp)   
                            # wTemp = np.reshape(wTemp, [1] + list(wTemp.shape) + [1])    
                            # if 'conv2' in layer.name: : now to save time
                            if len(layer_weights.shape) == 2: # dense
                                orig_shape = tf.shape(layer_weights).numpy()
                                stp = 50
                                while np.ceil(orig_shape[0]/stp) > 60:
                                    stp = stp*2
                                wt_temp = tf.reshape(layer_weights[0:orig_shape[0]:stp, 0:orig_shape[1]:stp], [int(np.ceil(orig_shape[0]/stp)), -1])
                                print(wt_temp.shape) 
                                
                            if len(layer_weights.shape) == 4: # conv2d
                                orig_shape = tf.shape(layer_weights).numpy()
                                stp = 50
                                while np.ceil(orig_shape[2]*orig_shape[3]/stp) > 60:
                                    stp = stp*2
                                print([np.prod(orig_shape[:2]), -1])
                                wt_temp = tf.reshape(layer_weights[:,:, 0:orig_shape[2]:stp, 0:orig_shape[3]:stp], [np.prod(orig_shape[:2]), -1])
                                print(wt_temp.shape) 

                            if wt_temp.shape[0] > wt_temp.shape[1]:
                                wt_temp = tf.transpose(wt_temp, perm=[1,0])
                            wTemp = plot_to_image(plot_weights(wt_temp))            
                            tf.summary.image(layer.name+'/kernel', wTemp, step=batch)    

                    # print(layer_weights.shape)
                    # print(self.polarityConfig[layer.name].shape)
                    checker = np.multiply(layer_weights, self.polarityConfig[layer.name])
                    if self.logFreeW:
                        with self._train_writer('_freeW').as_default():
                            tf.summary.histogram(layer.name+'/checker', checker, step=batch)

                    layer_weights = self.check_polarity(layer_weights, checker)

                    if self.logResetW:
                        with self._train_writer('').as_default():
                            tf.summary.histogram(layer.name+'/checker', np.multiply(layer_weights, self.polarityConfig[layer.name]), step=batch)
                    layer.set_weights([layer_weights, layer_bias])

                if self.logResetW:
                    with self._train_writer('').as_default():
                        tf.summary.histogram(layer.name+'/kernel', layer_weights, step=batch)
                        tf.summary.histogram(layer.name+'/bias', layer_bias, step=batch)
                        # wTemp = tf.cast(layer_weights, dtype = tf.float32)      
                        # wTemp = self.normalize_tensor(wTemp)                          
                        # wTemp = np.reshape(wTemp, [1] + list(wTemp.shape) + [1]) 
                        # if 'conv2' in layer.name: : now to save time
                        if len(layer_weights.shape) == 2: # dense
                            orig_shape = tf.shape(layer_weights).numpy()
                            stp = 50
                            while np.ceil(orig_shape[0]/stp) > 60:
                                stp = stp*2
                            wt_temp = tf.reshape(layer_weights[0:orig_shape[0]:stp, 0:orig_shape[1]:stp], [int(np.ceil(orig_shape[0]/stp)), -1])
                            print(wt_temp.shape) 
                            
                        if len(layer_weights.shape) == 4: # conv2d
                            orig_shape = tf.shape(layer_weights).numpy()
                            stp = 50
                            while np.ceil(orig_shape[2]*orig_shape[3]/stp) > 60:
                                stp = stp*2
                            print([np.prod(orig_shape[:2]), -1])
                            wt_temp = tf.reshape(layer_weights[:,:, 0:orig_shape[2]:stp, 0:orig_shape[3]:stp], [np.prod(orig_shape[:2]), -1])
                            print(wt_temp.shape) 
                            
                        if wt_temp.shape[0] > wt_temp.shape[1]:
                            wt_temp = tf.transpose(wt_temp, perm=[1,0])
                        wTemp = plot_to_image(plot_weights(wt_temp)) 
                        tf.summary.image(layer.name+'/kernel', wTemp, step=batch) 

        if self.logResetW:
            train_logs = self.model.evaluate(self.train_data)
            train_logs = {'loss': train_logs[0], 'accuracy': train_logs[1]}
            logs.update(train_logs)
            val_logs = self.model.evaluate(self.validation_data)
            val_logs = {'val_loss': val_logs[0], 'val_accuracy': val_logs[1]}
            logs.update(val_logs) 
            self._log_batch_metrics(batch, logs, '')
        # print(logs) # won't log here, will log in the tensorboard callback   
         
        for layer in self.model.layers:
            layer.trainable = True    
    
    def on_train_end(self, logs=None):
        # if self.logResetW | self.logFreeW:
        # will always close it as always want to write the initialization accuracy - see if it is too slow
        for writer in self._writers.values():
            writer.close()