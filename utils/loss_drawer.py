import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from matplotlib.ticker import NullFormatter
#获取今天的字符串
def getToday():
    return time.strftime("%Y-%m-%d %H-%M-%S",time.localtime(time.time()))

def LossSaver(lists):
    with open("{}.txt".format(getToday()),'a') as file:
        lists = [str(line)+"\n" for line in lists]
        file.writelines(lists)
    print("Saving the loss list for pic.")
        
def LossLoader(file):
    with open(file,'r') as file:
        Loss_list = []
        for line in file.readlines():    
            Loss_list.append(eval(line))
    return Loss_list


def LossDrawer(Loss_list, k=0):
    y_all = [i[0] for i in Loss_list]
    y_ptr = [i[1] for i in Loss_list]
    y_opr = [i[2] for i in Loss_list]
    y_dom = [i[3] for i in Loss_list]
    y_cfm = [i[4] for i in Loss_list]
    lr = [i[5] for i in Loss_list]
    
    N = len(Loss_list)
    x = range(0, N)
    #plt.subplot(1, 1, 1)
    #y_all, x = y_all[k:], x[k:]
    plt.plot(x[k:], lr[k:], 'r--', color='black', linewidth=1)
    plt.title('all loss')
    plt.ylabel('L')
    ax = plt.gca()
    #ax.set_yscale('logit')
    #.set_yticklabels(["20", "200", "500"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    plt.grid(True)
    '''
    plt.plot(x, y_ptr, '.-', color='red')
    plt.title('ptr loss')
    plt.ylabel('Lptr')
    
    plt.plot(x, y_opr, '.-', color='blue')
    plt.title('opr loss')
    plt.ylabel('Lopr')
    
    plt.plot(x, y_dom, '.-', color='green')
    plt.title('dom loss')
    plt.ylabel('Ldom')
    '''
    plt.show()
    #plt.savefig("{}.jpg".format(getToday()), dpi=300)


if __name__ == '__main__':

    Loss_list = LossLoader("2020-04-11 19-22-41.txt")
    #print(Loss_list)
    LossDrawer(Loss_list, k =0)




