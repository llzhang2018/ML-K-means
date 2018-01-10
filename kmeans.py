#coding=utf-8

import scipy.io as sio
from numpy import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.image as mpimg
import random
#===================================加载数据============================#
data=sio.loadmat('C:\Users\DELL\Desktop\\data1.mat')
data=data['X']
print data.shape

#===================================寻找聚类中心========================#
def findClosestCentroids(X, centroids):
	m,n=X.shape                     #获取样本规模
	K=centroids.shape[0]            #中心个数
	idx=zeros([m,1])                #每个样本对应一个中心（1-K）	
	for i in range(m):     
		d=zeros([K,1])
		tmp=zeros([1,n])
		position=0
		for j in range(K):
			tmp=X[i]-centroids[j]
			d[j]=tmp.dot(tmp.T)     #计算当前样本距离每个中心的距离
		d_min=min(d)                #position为1-K中某个值
		for j in range(K):
			if d[j]==d_min:
				position=j				
		idx[i]=position
	return idx

#====================================计算聚类中心位置======================#
def computeCentroids(X, idx, K):
	m,n=X.shape
	centroids=zeros([K,n])
	for i in range(K):
		iter=0
		tmp=zeros([1,n])
		for j in range(m):			
			if idx[j]==i:           #将属于聚类i的向量累加
				iter+=1
				tmp=tmp+X[j]
		centroids[i]=(1.0/iter)*tmp #求得新中心的位置
	return centroids

#====================================随机选取K个聚类中心====================#
def initCentroids(X,K):
	m,n=X.shape
	arr=range(m)
	idx=random.sample(arr,K)        #从arr中选取K个不重复样本
	#print idx
	centroids=zeros([K,n])
	for i in range(len(idx)):
		centroids[i]=X[idx[i]]
	return centroids
	
#====================================K-means主循环==========================#
def runkMeans(X, initial_centroids, max_iters):
	m,n=X.shape
	K=initial_centroids.shape[0]
	idx=zeros([m,1])
	centroids=initial_centroids
	
	for i in range(max_iters):      #迭代max_iters
		idx=findClosestCentroids(X,centroids)
		centroids=computeCentroids(X,idx,K)
	return centroids
	
#====================================功能测试==============================#
K=3
initial_centroids=array([[3,3],[6,2],[8,5]])
idx=findClosestCentroids(data,initial_centroids)
#print idx[:3]

centroids = computeCentroids(data, idx, K)
print('Centroids computed after initial finding of closest centroids: ')
print centroids

Max_iter=10
centroids=runkMeans(data,initial_centroids,Max_iter)
print('Centroids computed after 10 iterations: ')
print centroids

#====================================图片压缩===============================#
A = mpimg.imread('C:\Users\DELL\Desktop\\bird_small.png') 
fig=plt.figure()
plt.subplot(121)
plt.imshow(A)

m,n,z=A.shape     #[128,128,3]
A=A/255.0         # 0-1之间
A=A.reshape(m*n,z)#重新排列为m*n维样本
K_img = 16;       #16个聚类中心
A_iters = 10;     #迭代10次
init_centroids=initCentroids(A,K_img)           #生成初始化聚类中心
A_centroids=runkMeans(A,init_centroids,A_iters) #执行K-means
print A_centroids.shape

A_idx=findClosestCentroids(A,init_centroids)

A_recover=zeros([m*n,z])
for i in range(len(A_idx)):
	A_recover[i,:]=A_centroids[int(A_idx[i])]

A_recover=A_recover.reshape(m,n,z)
A_recover=A_recover*255
print A_recover.shape
plt.subplot(122)
plt.imshow(A_recover)
plt.show()