from Mnist import Mnist
from Network import Network

def start():
    m = Mnist('train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
              't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
    net = Network([784, 30, 10])
    net.SGD(m.train_data, 30, 10, 3.0, test_data=m.test_data)

if __name__ == '__main__':
    start()