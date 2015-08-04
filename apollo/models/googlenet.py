import os

from apollo import layers

def weights_file():
    filename = 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
    if not os.path.exists(filename):
        raise OSError('Please download the GoogLeNet model first with \
./scripts/download_model_binary.py models/bvlc_googlenet')
    return filename

def googlenet_layers():
    weight_filler = layers.Filler(type="xavier")
    bias_filler = layers.Filler(type="constant", value=0.2)
    conv_lr_mults = [1.0, 2.0]
    conv_decay_mults = [1.0, 0.0]

    googlenet_layers = [
        layers.Convolution(name="conv1/7x7_s2", bottoms=["data"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=7, stride=2, pad=3, weight_filler=weight_filler, bias_filler=bias_filler, num_output=64),
        layers.ReLU(name="conv1/relu_7x7", bottoms=["conv1/7x7_s2"], tops=["conv1/7x7_s2"]),
        layers.Pooling(name="pool1/3x3_s2", bottoms=["conv1/7x7_s2"], kernel_size=3, stride=2),
        layers.LRN(name="pool1/norm1", bottoms=["pool1/3x3_s2"], local_size=5, alpha=0.0001, beta=0.75),
        layers.Convolution(name="conv2/3x3_reduce", bottoms=["pool1/norm1"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=64),
        layers.ReLU(name="conv2/relu_3x3_reduce", bottoms=["conv2/3x3_reduce"], tops=["conv2/3x3_reduce"]),
        layers.Convolution(name="conv2/3x3", bottoms=["conv2/3x3_reduce"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=3, pad=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=192),
        layers.ReLU(name="conv2/relu_3x3", bottoms=["conv2/3x3"], tops=["conv2/3x3"]),
        layers.LRN(name="conv2/norm2", bottoms=["conv2/3x3"], local_size=5, alpha=0.0001, beta=0.75),
        layers.Pooling(name="pool2/3x3_s2", bottoms=["conv2/norm2"], kernel_size=3, stride=2),
        layers.Convolution(name="inception_3a/1x1", bottoms=["pool2/3x3_s2"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=64),
        layers.ReLU(name="inception_3a/relu_1x1", bottoms=["inception_3a/1x1"], tops=["inception_3a/1x1"]),
        layers.Convolution(name="inception_3a/3x3_reduce", bottoms=["pool2/3x3_s2"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=96),
        layers.ReLU(name="inception_3a/relu_3x3_reduce", bottoms=["inception_3a/3x3_reduce"], tops=["inception_3a/3x3_reduce"]),
        layers.Convolution(name="inception_3a/3x3", bottoms=["inception_3a/3x3_reduce"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=3, pad=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=128),
        layers.ReLU(name="inception_3a/relu_3x3", bottoms=["inception_3a/3x3"], tops=["inception_3a/3x3"]),
        layers.Convolution(name="inception_3a/5x5_reduce", bottoms=["pool2/3x3_s2"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=16),
        layers.ReLU(name="inception_3a/relu_5x5_reduce", bottoms=["inception_3a/5x5_reduce"], tops=["inception_3a/5x5_reduce"]),
        layers.Convolution(name="inception_3a/5x5", bottoms=["inception_3a/5x5_reduce"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=5, pad=2, weight_filler=weight_filler, bias_filler=bias_filler, num_output=32),
        layers.ReLU(name="inception_3a/relu_5x5", bottoms=["inception_3a/5x5"], tops=["inception_3a/5x5"]),
        layers.Pooling(name="inception_3a/pool", bottoms=["pool2/3x3_s2"], kernel_size=3, stride=1, pad=1),
        layers.Convolution(name="inception_3a/pool_proj", bottoms=["inception_3a/pool"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=32),
        layers.ReLU(name="inception_3a/relu_pool_proj", bottoms=["inception_3a/pool_proj"], tops=["inception_3a/pool_proj"]),
        layers.Concat(name="inception_3a/output", bottoms=["inception_3a/1x1", "inception_3a/3x3", "inception_3a/5x5", "inception_3a/pool_proj"]),
        layers.Convolution(name="inception_3b/1x1", bottoms=["inception_3a/output"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=128),
        layers.ReLU(name="inception_3b/relu_1x1", bottoms=["inception_3b/1x1"], tops=["inception_3b/1x1"]),
        layers.Convolution(name="inception_3b/3x3_reduce", bottoms=["inception_3a/output"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=128),
        layers.ReLU(name="inception_3b/relu_3x3_reduce", bottoms=["inception_3b/3x3_reduce"], tops=["inception_3b/3x3_reduce"]),
        layers.Convolution(name="inception_3b/3x3", bottoms=["inception_3b/3x3_reduce"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=3, pad=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=192),
        layers.ReLU(name="inception_3b/relu_3x3", bottoms=["inception_3b/3x3"], tops=["inception_3b/3x3"]),
        layers.Convolution(name="inception_3b/5x5_reduce", bottoms=["inception_3a/output"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=32),
        layers.ReLU(name="inception_3b/relu_5x5_reduce", bottoms=["inception_3b/5x5_reduce"], tops=["inception_3b/5x5_reduce"]),
        layers.Convolution(name="inception_3b/5x5", bottoms=["inception_3b/5x5_reduce"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=5, pad=2, weight_filler=weight_filler, bias_filler=bias_filler, num_output=96),
        layers.ReLU(name="inception_3b/relu_5x5", bottoms=["inception_3b/5x5"], tops=["inception_3b/5x5"]),
        layers.Pooling(name="inception_3b/pool", bottoms=["inception_3a/output"], kernel_size=3, stride=1, pad=1),
        layers.Convolution(name="inception_3b/pool_proj", bottoms=["inception_3b/pool"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=64),
        layers.ReLU(name="inception_3b/relu_pool_proj", bottoms=["inception_3b/pool_proj"], tops=["inception_3b/pool_proj"]),
        layers.Concat(name="inception_3b/output", bottoms=["inception_3b/1x1", "inception_3b/3x3", "inception_3b/5x5", "inception_3b/pool_proj"]),
        layers.Pooling(name="pool3/3x3_s2", bottoms=["inception_3b/output"], kernel_size=3, stride=2),
        layers.Convolution(name="inception_4a/1x1", bottoms=["pool3/3x3_s2"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=192),
        layers.ReLU(name="inception_4a/relu_1x1", bottoms=["inception_4a/1x1"], tops=["inception_4a/1x1"]),
        layers.Convolution(name="inception_4a/3x3_reduce", bottoms=["pool3/3x3_s2"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=96),
        layers.ReLU(name="inception_4a/relu_3x3_reduce", bottoms=["inception_4a/3x3_reduce"], tops=["inception_4a/3x3_reduce"]),
        layers.Convolution(name="inception_4a/3x3", bottoms=["inception_4a/3x3_reduce"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=3, pad=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=208),
        layers.ReLU(name="inception_4a/relu_3x3", bottoms=["inception_4a/3x3"], tops=["inception_4a/3x3"]),
        layers.Convolution(name="inception_4a/5x5_reduce", bottoms=["pool3/3x3_s2"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=16),
        layers.ReLU(name="inception_4a/relu_5x5_reduce", bottoms=["inception_4a/5x5_reduce"], tops=["inception_4a/5x5_reduce"]),
        layers.Convolution(name="inception_4a/5x5", bottoms=["inception_4a/5x5_reduce"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=5, pad=2, weight_filler=weight_filler, bias_filler=bias_filler, num_output=48),
        layers.ReLU(name="inception_4a/relu_5x5", bottoms=["inception_4a/5x5"], tops=["inception_4a/5x5"]),
        layers.Pooling(name="inception_4a/pool", bottoms=["pool3/3x3_s2"], kernel_size=3, stride=1, pad=1),
        layers.Convolution(name="inception_4a/pool_proj", bottoms=["inception_4a/pool"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=64),
        layers.ReLU(name="inception_4a/relu_pool_proj", bottoms=["inception_4a/pool_proj"], tops=["inception_4a/pool_proj"]),
        layers.Concat(name="inception_4a/output", bottoms=["inception_4a/1x1", "inception_4a/3x3", "inception_4a/5x5", "inception_4a/pool_proj"]),
        layers.Pooling(name="loss1/ave_pool", bottoms=["inception_4a/output"], kernel_size=5, stride=3, pool='AVE'),
        layers.Convolution(name="loss1/conv", bottoms=["loss1/ave_pool"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=128),
        layers.ReLU(name="loss1/relu_conv", bottoms=["loss1/conv"], tops=["loss1/conv"]),
        layers.InnerProduct(name="loss1/fc", bottoms=["loss1/conv"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, weight_filler=weight_filler, bias_filler=bias_filler, num_output=1024),
        layers.ReLU(name="loss1/relu_fc", bottoms=["loss1/fc"], tops=["loss1/fc"]),
        layers.Dropout(name="loss1/drop_fc", bottoms=["loss1/fc"], tops=["loss1/fc"], dropout_ratio=0.7, phase='TRAIN'),
        layers.InnerProduct(name="loss1/classifier", bottoms=["loss1/fc"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, weight_filler=weight_filler, bias_filler=layers.Filler(type="constant", value=0.0), num_output=1000),
        layers.SoftmaxWithLoss(name="loss1/loss", bottoms=["loss1/classifier", "label"], tops=["loss1/loss1"], loss_weight=0.3),
        layers.Convolution(name="inception_4b/1x1", bottoms=["inception_4a/output"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=160),
        layers.ReLU(name="inception_4b/relu_1x1", bottoms=["inception_4b/1x1"], tops=["inception_4b/1x1"]),
        layers.Convolution(name="inception_4b/3x3_reduce", bottoms=["inception_4a/output"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=112),
        layers.ReLU(name="inception_4b/relu_3x3_reduce", bottoms=["inception_4b/3x3_reduce"], tops=["inception_4b/3x3_reduce"]),
        layers.Convolution(name="inception_4b/3x3", bottoms=["inception_4b/3x3_reduce"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=3, pad=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=224),
        layers.ReLU(name="inception_4b/relu_3x3", bottoms=["inception_4b/3x3"], tops=["inception_4b/3x3"]),
        layers.Convolution(name="inception_4b/5x5_reduce", bottoms=["inception_4a/output"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=24),
        layers.ReLU(name="inception_4b/relu_5x5_reduce", bottoms=["inception_4b/5x5_reduce"], tops=["inception_4b/5x5_reduce"]),
        layers.Convolution(name="inception_4b/5x5", bottoms=["inception_4b/5x5_reduce"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=5, pad=2, weight_filler=weight_filler, bias_filler=bias_filler, num_output=64),
        layers.ReLU(name="inception_4b/relu_5x5", bottoms=["inception_4b/5x5"], tops=["inception_4b/5x5"]),
        layers.Pooling(name="inception_4b/pool", bottoms=["inception_4a/output"], kernel_size=3, stride=1, pad=1),
        layers.Convolution(name="inception_4b/pool_proj", bottoms=["inception_4b/pool"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=64),
        layers.ReLU(name="inception_4b/relu_pool_proj", bottoms=["inception_4b/pool_proj"], tops=["inception_4b/pool_proj"]),
        layers.Concat(name="inception_4b/output", bottoms=["inception_4b/1x1", "inception_4b/3x3", "inception_4b/5x5", "inception_4b/pool_proj"]),
        layers.Convolution(name="inception_4c/1x1", bottoms=["inception_4b/output"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=128),
        layers.ReLU(name="inception_4c/relu_1x1", bottoms=["inception_4c/1x1"], tops=["inception_4c/1x1"]),
        layers.Convolution(name="inception_4c/3x3_reduce", bottoms=["inception_4b/output"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=128),
        layers.ReLU(name="inception_4c/relu_3x3_reduce", bottoms=["inception_4c/3x3_reduce"], tops=["inception_4c/3x3_reduce"]),
        layers.Convolution(name="inception_4c/3x3", bottoms=["inception_4c/3x3_reduce"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=3, pad=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=256),
        layers.ReLU(name="inception_4c/relu_3x3", bottoms=["inception_4c/3x3"], tops=["inception_4c/3x3"]),
        layers.Convolution(name="inception_4c/5x5_reduce", bottoms=["inception_4b/output"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=24),
        layers.ReLU(name="inception_4c/relu_5x5_reduce", bottoms=["inception_4c/5x5_reduce"], tops=["inception_4c/5x5_reduce"]),
        layers.Convolution(name="inception_4c/5x5", bottoms=["inception_4c/5x5_reduce"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=5, pad=2, weight_filler=weight_filler, bias_filler=bias_filler, num_output=64),
        layers.ReLU(name="inception_4c/relu_5x5", bottoms=["inception_4c/5x5"], tops=["inception_4c/5x5"]),
        layers.Pooling(name="inception_4c/pool", bottoms=["inception_4b/output"], kernel_size=3, stride=1, pad=1),
        layers.Convolution(name="inception_4c/pool_proj", bottoms=["inception_4c/pool"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=64),
        layers.ReLU(name="inception_4c/relu_pool_proj", bottoms=["inception_4c/pool_proj"], tops=["inception_4c/pool_proj"]),
        layers.Concat(name="inception_4c/output", bottoms=["inception_4c/1x1", "inception_4c/3x3", "inception_4c/5x5", "inception_4c/pool_proj"]),
        layers.Convolution(name="inception_4d/1x1", bottoms=["inception_4c/output"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=112),
        layers.ReLU(name="inception_4d/relu_1x1", bottoms=["inception_4d/1x1"], tops=["inception_4d/1x1"]),
        layers.Convolution(name="inception_4d/3x3_reduce", bottoms=["inception_4c/output"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=144),
        layers.ReLU(name="inception_4d/relu_3x3_reduce", bottoms=["inception_4d/3x3_reduce"], tops=["inception_4d/3x3_reduce"]),
        layers.Convolution(name="inception_4d/3x3", bottoms=["inception_4d/3x3_reduce"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=3, pad=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=288),
        layers.ReLU(name="inception_4d/relu_3x3", bottoms=["inception_4d/3x3"], tops=["inception_4d/3x3"]),
        layers.Convolution(name="inception_4d/5x5_reduce", bottoms=["inception_4c/output"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=32),
        layers.ReLU(name="inception_4d/relu_5x5_reduce", bottoms=["inception_4d/5x5_reduce"], tops=["inception_4d/5x5_reduce"]),
        layers.Convolution(name="inception_4d/5x5", bottoms=["inception_4d/5x5_reduce"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=5, pad=2, weight_filler=weight_filler, bias_filler=bias_filler, num_output=64),
        layers.ReLU(name="inception_4d/relu_5x5", bottoms=["inception_4d/5x5"], tops=["inception_4d/5x5"]),
        layers.Pooling(name="inception_4d/pool", bottoms=["inception_4c/output"], kernel_size=3, stride=1, pad=1),
        layers.Convolution(name="inception_4d/pool_proj", bottoms=["inception_4d/pool"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=64),
        layers.ReLU(name="inception_4d/relu_pool_proj", bottoms=["inception_4d/pool_proj"], tops=["inception_4d/pool_proj"]),
        layers.Concat(name="inception_4d/output", bottoms=["inception_4d/1x1", "inception_4d/3x3", "inception_4d/5x5", "inception_4d/pool_proj"]),
        layers.Pooling(name="loss2/ave_pool", bottoms=["inception_4d/output"], kernel_size=5, stride=3, pool='AVE'),
        layers.Convolution(name="loss2/conv", bottoms=["loss2/ave_pool"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=128),
        layers.ReLU(name="loss2/relu_conv", bottoms=["loss2/conv"], tops=["loss2/conv"]),
        layers.InnerProduct(name="loss2/fc", bottoms=["loss2/conv"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, weight_filler=weight_filler, bias_filler=bias_filler, num_output=1024),
        layers.ReLU(name="loss2/relu_fc", bottoms=["loss2/fc"], tops=["loss2/fc"]),
        layers.Dropout(name="loss2/drop_fc", bottoms=["loss2/fc"], tops=["loss2/fc"], dropout_ratio=0.7, phase='TRAIN'),
        layers.InnerProduct(name="loss2/classifier", bottoms=["loss2/fc"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, weight_filler=weight_filler, bias_filler=layers.Filler(type="constant", value=0.0), num_output=1000),
        layers.SoftmaxWithLoss(name="loss2/loss", bottoms=["loss2/classifier", "label"], tops=["loss2/loss1"], loss_weight=0.3),
        layers.Convolution(name="inception_4e/1x1", bottoms=["inception_4d/output"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=256),
        layers.ReLU(name="inception_4e/relu_1x1", bottoms=["inception_4e/1x1"], tops=["inception_4e/1x1"]),
        layers.Convolution(name="inception_4e/3x3_reduce", bottoms=["inception_4d/output"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=160),
        layers.ReLU(name="inception_4e/relu_3x3_reduce", bottoms=["inception_4e/3x3_reduce"], tops=["inception_4e/3x3_reduce"]),
        layers.Convolution(name="inception_4e/3x3", bottoms=["inception_4e/3x3_reduce"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=3, pad=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=320),
        layers.ReLU(name="inception_4e/relu_3x3", bottoms=["inception_4e/3x3"], tops=["inception_4e/3x3"]),
        layers.Convolution(name="inception_4e/5x5_reduce", bottoms=["inception_4d/output"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=32),
        layers.ReLU(name="inception_4e/relu_5x5_reduce", bottoms=["inception_4e/5x5_reduce"], tops=["inception_4e/5x5_reduce"]),
        layers.Convolution(name="inception_4e/5x5", bottoms=["inception_4e/5x5_reduce"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=5, pad=2, weight_filler=weight_filler, bias_filler=bias_filler, num_output=128),
        layers.ReLU(name="inception_4e/relu_5x5", bottoms=["inception_4e/5x5"], tops=["inception_4e/5x5"]),
        layers.Pooling(name="inception_4e/pool", bottoms=["inception_4d/output"], kernel_size=3, stride=1, pad=1),
        layers.Convolution(name="inception_4e/pool_proj", bottoms=["inception_4e/pool"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=128),
        layers.ReLU(name="inception_4e/relu_pool_proj", bottoms=["inception_4e/pool_proj"], tops=["inception_4e/pool_proj"]),
        layers.Concat(name="inception_4e/output", bottoms=["inception_4e/1x1", "inception_4e/3x3", "inception_4e/5x5", "inception_4e/pool_proj"]),
        layers.Pooling(name="pool4/3x3_s2", bottoms=["inception_4e/output"], kernel_size=3, stride=2),
        layers.Convolution(name="inception_5a/1x1", bottoms=["pool4/3x3_s2"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=256),
        layers.ReLU(name="inception_5a/relu_1x1", bottoms=["inception_5a/1x1"], tops=["inception_5a/1x1"]),
        layers.Convolution(name="inception_5a/3x3_reduce", bottoms=["pool4/3x3_s2"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=160),
        layers.ReLU(name="inception_5a/relu_3x3_reduce", bottoms=["inception_5a/3x3_reduce"], tops=["inception_5a/3x3_reduce"]),
        layers.Convolution(name="inception_5a/3x3", bottoms=["inception_5a/3x3_reduce"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=3, pad=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=320),
        layers.ReLU(name="inception_5a/relu_3x3", bottoms=["inception_5a/3x3"], tops=["inception_5a/3x3"]),
        layers.Convolution(name="inception_5a/5x5_reduce", bottoms=["pool4/3x3_s2"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=32),
        layers.ReLU(name="inception_5a/relu_5x5_reduce", bottoms=["inception_5a/5x5_reduce"], tops=["inception_5a/5x5_reduce"]),
        layers.Convolution(name="inception_5a/5x5", bottoms=["inception_5a/5x5_reduce"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=5, pad=2, weight_filler=weight_filler, bias_filler=bias_filler, num_output=128),
        layers.ReLU(name="inception_5a/relu_5x5", bottoms=["inception_5a/5x5"], tops=["inception_5a/5x5"]),
        layers.Pooling(name="inception_5a/pool", bottoms=["pool4/3x3_s2"], kernel_size=3, stride=1, pad=1),
        layers.Convolution(name="inception_5a/pool_proj", bottoms=["inception_5a/pool"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=128),
        layers.ReLU(name="inception_5a/relu_pool_proj", bottoms=["inception_5a/pool_proj"], tops=["inception_5a/pool_proj"]),
        layers.Concat(name="inception_5a/output", bottoms=["inception_5a/1x1", "inception_5a/3x3", "inception_5a/5x5", "inception_5a/pool_proj"]),
        layers.Convolution(name="inception_5b/1x1", bottoms=["inception_5a/output"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=384),
        layers.ReLU(name="inception_5b/relu_1x1", bottoms=["inception_5b/1x1"], tops=["inception_5b/1x1"]),
        layers.Convolution(name="inception_5b/3x3_reduce", bottoms=["inception_5a/output"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=192),
        layers.ReLU(name="inception_5b/relu_3x3_reduce", bottoms=["inception_5b/3x3_reduce"], tops=["inception_5b/3x3_reduce"]),
        layers.Convolution(name="inception_5b/3x3", bottoms=["inception_5b/3x3_reduce"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=3, pad=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=384),
        layers.ReLU(name="inception_5b/relu_3x3", bottoms=["inception_5b/3x3"], tops=["inception_5b/3x3"]),
        layers.Convolution(name="inception_5b/5x5_reduce", bottoms=["inception_5a/output"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=48),
        layers.ReLU(name="inception_5b/relu_5x5_reduce", bottoms=["inception_5b/5x5_reduce"], tops=["inception_5b/5x5_reduce"]),
        layers.Convolution(name="inception_5b/5x5", bottoms=["inception_5b/5x5_reduce"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=5, pad=2, weight_filler=weight_filler, bias_filler=bias_filler, num_output=128),
        layers.ReLU(name="inception_5b/relu_5x5", bottoms=["inception_5b/5x5"], tops=["inception_5b/5x5"]),
        layers.Pooling(name="inception_5b/pool", bottoms=["inception_5a/output"], kernel_size=3, stride=1, pad=1),
        layers.Convolution(name="inception_5b/pool_proj", bottoms=["inception_5b/pool"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, kernel_size=1, weight_filler=weight_filler, bias_filler=bias_filler, num_output=128),
        layers.ReLU(name="inception_5b/relu_pool_proj", bottoms=["inception_5b/pool_proj"], tops=["inception_5b/pool_proj"]),
        layers.Concat(name="inception_5b/output", bottoms=["inception_5b/1x1", "inception_5b/3x3", "inception_5b/5x5", "inception_5b/pool_proj"]),
        layers.Pooling(name="pool5/7x7_s1", bottoms=["inception_5b/output"], kernel_size=7, stride=1, pool='AVE'),
        layers.Dropout(name="pool5/drop_7x7_s1", bottoms=["pool5/7x7_s1"], tops=["pool5/7x7_s1"], dropout_ratio=0.4, phase='TRAIN'),
        layers.InnerProduct(name="loss3/classifier", bottoms=["pool5/7x7_s1"], param_lr_mults=conv_lr_mults, param_decay_mults=conv_decay_mults, weight_filler=weight_filler, bias_filler=layers.Filler(type="constant", value=0.0), num_output=1000),
        layers.SoftmaxWithLoss(name="loss3/loss3", bottoms=["loss3/classifier", "label"], loss_weight=1.0),
    ]
    return googlenet_layers
