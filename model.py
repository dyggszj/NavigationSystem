import tensorflow as tf
from utils import *
import numpy as np
np.set_printoptions(threshold=np.inf)
def schedule(epoch):
    print(epoch)
    if epoch < 50:
        return .1
    elif (epoch < 100):
        return .01
    elif epoch < 200:
        return .001
    elif epoch < 300:
        return .0001
    else:
        return .0000001      

#将LearningRateScheduler类实例化   
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

#将所有callback类放入，这里只有这一个类
callbacks_list = [lr_scheduler, ]


class Model():
    def __init__(self, *args):
        self.hasshow = False
        # x = tf.random.normal((1,9))                 #   模拟样本数据
        # print(x)
        self.model = tf.keras.Sequential([               #   定义全连接层结构
            tf.keras.layers.BatchNormalization(axis=1 ),
            
            tf.keras.layers.Dense(9,activation='relu'),  
            tf.keras.layers.Dense(3)                                    #   输出层不需要激活函数
        ])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer='adam',
                    loss=loss_fn,
                    metrics=['accuracy'])


        # out = self.model(x).numpy() 
        # myPrint("out",out)
        # out = tf.nn.softmax(out).numpy()     
        # loss = loss_fn(0, out).numpy()
        # myPrint('loss',loss)

        # # print('out:',out)
        
        
    def loaddata(self):
        raw_left = readXlsx("/data/zhoutianyi/direction/data/RawData2/LeftTurn.xlsx")
        raw_right = readXlsx("/data/zhoutianyi/direction/data/RawData2/RightTurn.xlsx")
        raw_straight = readXlsx("/data/zhoutianyi/direction/data/RawData2/Straight.xlsx")

        left = np.array(raw_left)
        left = left[1:,:9]
        myPrint("leftshape",left.shape)#284行
        right = np.array(raw_right)
        right = right[1:,:9]
        myPrint("right",right.shape)#297行
        straight = np.array(raw_straight)
        straight = straight[1:,:9]
        myPrint("straight",straight.shape)#178行
        totalnum = straight.shape[0]+right.shape[0]+left.shape[0]
        myPrint("totol num ofdata",totalnum)

        x = np.concatenate((left,right,straight),axis=0)
        nums =  x.shape[0]
        myPrint("x",x.shape)
        
        y = np.zeros(nums)
        y[:left.shape[0]] = 0
        y[left.shape[0]:(left.shape[0]+right.shape[0])] = 1
        y[(left.shape[0]+right.shape[0]):(left.shape[0]+right.shape[0]+straight.shape[0])] = 2
        myPrint("y",y.shape)

        state = np.random.get_state()
        np.random.shuffle(x)
        np.random.set_state(state)
        np.random.shuffle(y)

        test_num = 30
        self.xtrain = x[:-test_num].astype("float64")
        self.ytrain = y[:-test_num].astype("float64")
        self.xtest = x[-test_num:].astype("float64")
        self.ytest = y[-test_num:].astype("float64")
        print(self.xtrain)
        print(self.ytrain)


    def loaddata_intergrate(self):
        raw_left = readXlsx("/data/zhoutianyi/direction/data/RawData2/LeftTurn.xlsx")
        raw_right = readXlsx("/data/zhoutianyi/direction/data/RawData2/RightTurn.xlsx")
        raw_straight = readXlsx("/data/zhoutianyi/direction/data/RawData2/Straight.xlsx")

        left = np.array(raw_left)
        left = left[1:,:9]
        myPrint("leftshape",left.shape)#284行
        right = np.array(raw_right)
        right = right[1:,:9]
        myPrint("right",right.shape)#297行
        straight = np.array(raw_straight)
        straight = straight[1:,:9]
        myPrint("straight",straight.shape)#178行
        totalnum = straight.shape[0]+right.shape[0]+left.shape[0]
        myPrint("totol num ofdata",totalnum)

        padding = 3
        left  = intergrate(left,padding)
        right  = intergrate(right,padding)
        straight  = intergrate(straight,padding)

        x = np.concatenate((left,right,straight),axis=0)
        nums =  x.shape[0]
        myPrint("x",x.shape)
        
        y = np.zeros(nums)
        y[:left.shape[0]] = 0
        y[left.shape[0]:(left.shape[0]+right.shape[0])] = 1
        y[(left.shape[0]+right.shape[0]):(left.shape[0]+right.shape[0]+straight.shape[0])] = 2
        myPrint("y",y.shape)

        state = np.random.get_state()
        np.random.shuffle(x)
        np.random.set_state(state)
        np.random.shuffle(y)

        test_num = 30
        self.xtrain = x[:-test_num].astype("float64")
        self.ytrain = y[:-test_num].astype("float64")
        self.xtest = x[-test_num:].astype("float64")
        self.ytest = y[-test_num:].astype("float64")
        print(self.xtrain)
        print(self.ytrain)
    
    def train(self):
        myPrint("self.xtrain",self.xtrain.shape)
        myPrint("self.ytrain",self.ytrain.shape)
        self.model.fit(self.xtrain,self.ytrain,epochs = 1)
        if(self.hasshow == False):
            self.hasshow = True
            print(self.model.summary()) 
            for i in self.model.trainable_variables: 
                print(i.name,i.shape)

        self.model.fit(self.xtrain,self.ytrain,epochs = 300,callbacks=callbacks_list)

    def eval(self):
        self.model.evaluate(self.xtest,  self.ytest, verbose=2)

    

if __name__ == "__main__":
    model = Model()
    model.loaddata_intergrate()
    model.train()
    model.eval()