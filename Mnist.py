from array import array
import struct
import numpy as np
class Mnist(object):
    def __init__(self,p1,p2,p3,p4):
        self.train_data=self.truncate(self.load_train_data(p1,p2),100,700)

        self.test_data=self.truncate(self.load_test_data(p3,p4),100,700)
        # self.truncate(100,700)
    def truncate(self,data,start,end):
        for i in range(len(data)):
            array=self.to_array(data[i][0])
            data[i]=(np.array([array[j] for j in range(start,end)]),data[i][1])

        # for i in range(len(self.test_data)):
        #     array = self.to_array(self.test_data[i][0])
        #     self.test_data[i] = (np.array([array[j] for j in range(start, end)]),self.test_data[i][1])
        return data
    # def to_array(self,index):
    #      to_array=lambda :return [i for  i in self.train_data[index][0]]
    def to_array(self,array):
        return [i for i in array]

    def load_train_data(self,p1,p2):
        try:
            label = open(p2)
        except Exception:
            print "FileNotFound"
            exit(-1)
        label.read(8)
        labels = list(array("B", label.read()))
        try:
            imgs=open(p1)
        except Exception:
            print "FileNotFound"
            exit(-1)
        magic, size, rows, cols = struct.unpack(">IIII", imgs.read(16))
        images = [float(i)/float(255) for i in list(array("B", imgs.read()))]

        images=[images[i:i+784] for i in xrange(0,len(images),784)]
        images = [np.reshape(x, (784, 1)) for x in images]
        training_data=zip(images,self.__vectorize_traning_labels(labels))

        return training_data

    def load_test_data(self,p3,p4):
        try:
            test_imgs=open(p3)
            test_labels=open(p4)
        except Exception:
            print "FileNotFound"
            exit(-1)
        magic, size, rows, cols = struct.unpack(">IIII", test_imgs.read(16))
        images = [float(i) / float(255) for i in list(array("B", test_imgs.read()))]

        images = [images[i:i + 784] for i in xrange(0, len(images), 784)]
        images = [np.reshape(x, (784, 1)) for x in images]
        test_labels.read(8)
        labels = list(array("B", test_labels.read()))
        return [(images[i],labels[i]) for i in xrange(len(labels))]






    def __vectorize_traning_labels(self,labels):
        vec=[np.array([[0] for i in range(10)]) for i in labels]
        for i in xrange(len(vec)):
            vec[i][labels[i]]=1
        return vec



