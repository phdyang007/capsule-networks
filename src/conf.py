class Config(object):
    # dataset = 'mnist', 'fashion' or 'cifar'
    dataset = 'cifar'
    data_size = [32, 32, 3]
    model_dir = './model/'
    img_dir = './img/'
    batch_size = 128
    learning_rate = 0.001
    training = 1000
    mnist_dir = '/research/byu2/shli5/data/mnist_data/'
    fashion_dir = '/research/byu2/shli5/data/fashion_data/'
    cifar_dir = '/research/byu2/shli5/data/cifar-10-python/cifar-10-batches-py'
    batch_capacity = 1000

