import numpy as np
import os

class Pictures:
    def __init__(self):
        #----data structure for storing
        self.train_data = {}
        self.val_data = {}
        self.test_data = {}
        #----setting base path
        self.path = os.path.abspath('.')
        try:
            self.data = open(self.path + '/' + 'train.txt')
            Path = []
            Tag = []
            for line in self.data:
                line = line.replace('\n', '').split(' ')
                Path.append(line[0])
                Tag.append(line[1])
            self.train_data['Path'] = Path
            self.train_data['Tag'] = Tag
            #print self.train_data['Tag']
        except Exception, e:
            print 'Reading training data error!'
            print e
        #----read val data
        try:
            self.data = open(self.path + '/' + 'val.txt')
            Path = []
            Tag = []
            for line in self.data:
                line = line.replace('\n', '').split(' ')
                Path.append(line[0])
                Tag.append(line[1])
            self.val_data['Path'] = Path
            self.val_data['Tag'] = Tag
        except Exception, e:
            print 'Reading val data error!'
            print e
        #----read test data
        try:
            self.data = open(self.path + '/' + 'test.txt')
            Path = []
            for line in self.data:
                line = line.replace('\n', '').split(' ')
                Path.append(line[0])
            self.test_data['Path'] = Path
        except Exception, e:
            print 'Reading test data error!'
            print e

    def __del__(self):
        self.data.close()

if __name__ == '__main__':
    ob = Pictures()