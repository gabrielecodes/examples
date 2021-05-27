import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import lax, vmap, jit, value_and_grad
from jax.nn import relu, softmax, logsumexp
from jax.nn.initializers import glorot_normal, normal
from jax import random
import time
import tensorflow_datasets as tfds


def serial(*layers):    
    """ sequence of network layers """
    n_layers = len(layers)
    init_funs, apply_funs = zip(*layers)
    def init_fun(key, input_shape):
        """ initializes all the layers """
        parameters = []
        keys = random.split(key, n_layers)
        for key, init_fun in zip(keys, init_funs):
            input_shape, params = init_fun(key, input_shape)
            parameters.append(params)
        return input_shape, parameters

    def apply_fun(parameters, inputs):
            for params, apply_fun in zip(parameters, apply_funs):
                inputs = apply_fun(params, inputs)
            return inputs

    return init_fun, apply_fun


def flatten_layer():
    """ adapts the output of a convolutional layer to the input of a dense layer """
    def init_fun(key, input_shape):
        out_shape = (input_shape[2] * input_shape[3], )
        W, b = np.empty(0), np.empty(0)
        return out_shape, (W, b)
        
    def apply_fun(params, in_tensor):                        
        flatten = in_tensor.reshape(in_tensor.shape[2] * in_tensor.shape[3], )
        return flatten

    return init_fun, apply_fun


def conv_layer(kernel_size, strides=(1, 1), padding='SAME', activation_fun=relu):
    """ convolutional layer implementation """
    def init_fun(key, input_shape):        
        k1, k2 = random.split(key, 2)    
        kernel_shape = (1, 1, kernel_size[0], kernel_size[1])        
        W = glorot_normal()(k1, kernel_shape)            
        b = normal(1e-3)(k2, (1,))
        out_shape = lax.conv_general_shape_tuple(input_shape,
                                                 kernel_shape,
                                                 strides, 
                                                 padding,
                                                 ('NCHW', 'OIHW', 'NCHW'))
        return out_shape, (W, b)

    def apply_fun(params, in_tensor):           
        W, b = params         
        if len(in_tensor.shape) == 3:
            in_tensor = in_tensor[jnp.newaxis, ...].astype(jnp.float32)
        return activation_fun(lax.conv(in_tensor, W, strides, padding) + b)

    return init_fun, apply_fun


def dense_layer(out_size, activation_fun):
    """ dense layer implementation """
    def init_fun(key, input_shape):
        k1, k2 = random.split(key, 2)           
        W = glorot_normal()(k1, (out_size, input_shape[0])) 
        b = random.normal(k2, (out_size, )) 
        out_shape = (out_size, )
        return out_shape, (W, b)

    def apply_fun(params, in_tensor):    
        W, b = params            
        if activation_fun.__name__ == "softmax":
            out = jnp.dot(W, in_tensor) + b
            return out - logsumexp(out)
        else:
            return activation_fun(jnp.dot(W, in_tensor) + b)
        
    return init_fun, apply_fun


def to_categorical(y, num_classes=10):
    return jnp.eye(num_classes, dtype='uint8')[y]


def accuracy(params, images, targets):        
    target_class = jnp.argmax(targets, axis=1)
    preds = batched_predict(params, images)
    predicted_class = jnp.argmax(preds, axis=1)        
    return jnp.mean(predicted_class == target_class)


def loss(params, images, targets):
    preds = batched_predict(params, images).squeeze()    
    return -jnp.mean(preds * targets) 


def update(params, x, y):               
    l, grads = value_and_grad(loss)(params, x, y)
    p = [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)]    
    return p, l


def get_train_batches(batch_size, data_dir):
    ds = tfds.load(name='mnist', split='train', as_supervised=True, data_dir=data_dir)
    batch = ds.batch(batch_size).prefetch(batch_size)
    return batch.as_numpy_iterator()


if __name__ == "__main__":
    data_dir = '/tmp/tfds'
    batch_size=128
    epochs = 10
    step_size = 0.02

    rng = random.PRNGKey(107)

    layers = [conv_layer(kernel_size=(3, 3), activation_fun=relu),
        conv_layer(kernel_size=(3, 3), activation_fun=relu),
        flatten_layer(),
        dense_layer(784, relu),
        dense_layer(512, relu),
        dense_layer(10, softmax)]

    in_shape = (batch_size, 1, 28, 28)
    init_network, model = serial(*layers)
    out_shape, net_params = init_network(rng, in_shape)
    batched_predict = vmap(model, in_axes=(None, 0))

    train_accuracies = []
    for epoch in range(epochs):
        start = time.time()
        for x, y in get_train_batches(batch_size, data_dir):        
            x = jnp.array(x).reshape(len(x), 1, 28, 28)
            y = to_categorical(jnp.array(y))            
            net_params, l = jit(update)(net_params, x, y)            
        
        train_acc = accuracy(net_params, x, y)    
        train_accuracies.append(train_acc)        
        print(f"Epoch {epoch + 1} Time = {time.time() - start:.3} secs.")     
        print(f"Loss = {l}")
        print(f"Training set accuracy {100 * train_acc:3.2f}")  
        print("")
    
    train_accuracies = []
    for epoch in range(epochs):
        start = time.time()
        for x, y in get_train_batches(batch_size, data_dir):        
            x = jnp.array(x).reshape(len(x), 1, 28, 28)
            y = to_categorical(jnp.array(y))            
            net_params, l = jit(update)(net_params, x, y)            
        
        train_acc = accuracy(net_params, x, y)    
        train_accuracies.append(train_acc)        
        print(f"Epoch {epoch + 1} Time = {time.time() - start:.3} secs.")     
        print(f"Loss = {l}")
        print(f"Training set accuracy {100 * train_acc:3.2f}")  
        print("")
    
    print("end")