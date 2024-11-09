import threading
import time

import numpy as np
import vg

import read_data
import matplotlib.pyplot as plt
from scipy.spatial import distance
from adjustText import adjust_text
from sklearn.metrics import accuracy_score, \
							confusion_matrix, \
							precision_score, \
							f1_score, \
							recall_score
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# tlabel=['Collecting-0' ,'bowing-1' ,'cleaning-2', 'looking-3', 'opening-4',
#  'passing-5' ,'picking-6', 'placing-7', 'pushing-8', 'reading-9' , 'sitting-10',
#  'standing-11' ,'standing_up-12', 'talking-13' ,'turing_front-14',
#  'turning-15' ,'walking-16']
# Helper function that combines the pattern layer and summation layer
dic = {'Collecting': 0, 'bowing': 1, 'cleaning': 2,  'looking': 3, 'opening': 4,
		 'passing': 5, 'picking': 6, 'placing': 7, 'pushing': 8, 'reading': 9, 'sitting': 10, 'standing': 11,
		 'standing_up': 12, 'talking': 13, 'turing_front': 14, 'turning': 15, 'walking': 16}
def gas(centre, x, sigma):
	centre = centre.reshape(1, -1)
	temp = -np.sum((centre - x) ** 2, axis = 1)
	temp = temp / (2 * sigma * sigma)
	temp = np.exp(temp)
	print(temp.shape)
	gaussian = np.sum(temp)

	return gaussian


def mgas(centre, x, sigma):
	centre = centre.reshape(1, -1)
	temp = -np.linalg.norm(centre-x,ord=1,axis=1)
	temp = temp / (2 * sigma * sigma)
	temp = np.exp(temp)
	gaussian = np.sum(temp)

	return gaussian
def elaplas(centre, x, sigma):
	centre = centre.reshape(1, -1)
	temp = -np.sum((centre - x) ** 2, axis=1)
	temp = temp /  sigma
	temp = np.exp(temp)
	gaussian = np.sum(temp)
	return gaussian

def laplas(centre, x, sigma):
	centre = centre.reshape(1, -1)
	temp = -np.linalg.norm(centre-x,ord=1,axis=1)
	# temp = -np.sum((centre - x) ** 2, axis=1)
	temp = temp /  sigma
	temp = np.exp(temp)
	gaussian = np.sum(temp)
	return gaussian
def colaplas(centre, x, sigma):
	centre = centre.reshape(1, -1)
	num = np.dot([centre], np.array(x).T)  # 向量点乘
	denom = np.linalg.norm(centre) * np.linalg.norm(x, axis=1)  # 求模长的乘积
	res = num / denom
	i = np.arccos(res)
	i = np.nan_to_num(i)
	tt = np.pi - i
	res = np.minimum(i, tt)
	# res[np.isneginf(res)] = 0
	# res[np.isneginf(res)] = 0
	temp = res
	temp = temp/sigma
	temp = np.exp(-temp)
	gaussian = np.sum(temp)
	return gaussian
# def cosdistance(centre, x, sigma):
# 	centre = centre.reshape(1, -1)
# 	num = np.dot([centre], np.array(x).T)  # 向量点乘
#
# 	denom = np.linalg.norm(centre) * np.linalg.norm(x, axis=1)  # 求模长的乘积
# 	res = num / denom
# 	res[np.isneginf(res)] = 0
# 	temp=res
#
# 	# for i in range (x.shape[0]):
# 	# 	temp[i]=centre.dot(x[i])/(np.linalg.norm(centre)*np.linalg.norm(x[i]))
# 	# temp = centre.dot(x)/(np.linalg.norm(centre)*np.linalg.norm(x))
# 	temp = temp / (2 * sigma * sigma)
# 	temp = np.exp(temp)
# 	gaussian = np.sum(temp)
# 	print(gaussian.shape)
# 	return gaussian
# def cosdistance(centre, x, sigma):
# 	centre = centre.reshape(1, -1)
# 	centre=centre.reshape(19,3)
# 	x=x.reshape(x.shape[0],19,3)
# 	t=np.array([(np.dot([centre[i]], np.array(x[:,i]).T) )/(np.linalg.norm(centre[i]) * np.linalg.norm(x[:,i], axis=1))for i in range(centre.shape[0])])
# 	t=np.arccos(t)
# 	# l=np.pi-t
# 	# t=np.minimum(l,t)
# 	t=np.nan_to_num(t)
# 	t=np.sum(t,axis=0)
# 	res=t/19
# 	# res[np.isneginf(res)] = 0
# 	temp = res
# 	# for i in range (x.shape[0]):
# 	# 	temp[i]=centre.dot(x[i])/(np.linalg.norm(centre)*np.linalg.norm(x[i]))
# 	# temp = centre.dot(x)/(np.linalg.norm(centre)*np.linalg.norm(x))
# 	temp = temp / (2 * sigma * sigma)
# 	temp = np.exp(-temp)
# 	gaussian = np.sum(temp)
# 	return gaussian
def cosdistance(centre, x, sigma):
	centre = centre.reshape(1, -1)

	num = np.dot([centre], np.array(x).T)  # 向量点乘

	denom = np.linalg.norm(centre) * np.linalg.norm(x, axis=1)  # 求模长的乘积
	res = num / denom
	# for i in res[0][0]:
	# 	if i <0:
	# 		print(i)
	# 		print(np.arccos(i))
	i=np.arccos(res)
	i = np.nan_to_num(i)
	i=i[0][0]
	tt=np.pi-i
	res=np.minimum(i,tt)
	# res[np.isneginf(res)] = 0
	temp = res
	# for i in range (x.shape[0]):
	# 	temp[i]=centre.dot(x[i])/(np.linalg.norm(centre)*np.linalg.norm(x[i]))
	# temp = centre.dot(x)/(np.linalg.norm(centre)*np.linalg.norm(x))
	temp = temp / (2 * sigma * sigma)
	temp = np.exp(-temp)
	gaussian = np.sum(temp)
	return gaussian

def subset_by_class(data, labels):

	x_train_subsets = []
	
	for l in labels:
		indices = np.where(data['y_train'] == l)
		x_train_subsets.append(data['x_train'][indices, :])

	return x_train_subsets

def action_task(label):
	print("the action---"+str(label)+"---is done")
	print("the robot command can put here!")
def PNN(data,sigma,tag):
	SkeletonConnectionMap = [[1, 0],
							 [2, 1],
							 [3, 2],
							 [4, 2],
							 [5, 4],
							 [6, 5],
							 [7, 6],
							 [8, 2],
							 [9, 8],
							 [10, 9],
							 [11, 10],
							 [12, 0],
							 [13, 12],
							 [14, 13],
							 [15, 0],
							 [16, 15],
							 [17, 16],
							 [18, 3],
							 ]
	num_testset = data['x_test'].shape[0]
	d=data['x_train'].shape[1]
	labels = np.unique(data['y_train'])
	num_class = len(labels)
	# Splits the training set into subsets where each subset contains data points from a particular class	
	x_train_subsets = subset_by_class(data, labels)	
	p=[len(i[0])/data['x_train'].shape[0] for i in x_train_subsets]
	within=0
	between=0
	for n, subset in enumerate(x_train_subsets):
		within+=p[n]*np.var(np.array(subset[0]))
		between+=p[n]*np.sum((np.mean(np.array(subset[0]),axis=0)-np.mean(data['x_train'],axis=0))**2)
	# Variable for storing the summation layer values from each class
	summation_layer = np.zeros(num_class)
	
	# Variable for storing the predictions formacro each test data point
	predictions = np.zeros(num_testset)
	ax = plt.subplot(1, 2, 1)

	fig = plt.subplot(1, 2, 2, projection='3d')
	fig.view_init(-90, 90)
	fig.set_xlabel('x')
	fig.set_ylabel('y')
	fig.set_zlabel('z')
	fig.set_xlim(-1.500, 1.500)
	fig.set_ylim(-1.000, 1.000)
	fig.set_zlim(-1.000, 1.000)
	nm=np.unique(data['y_train'])
	print(nm,11111)
	for i, test_point in enumerate(data['x_test']):
		joints = test_point.reshape([19, 3])
		for j, subset in enumerate(x_train_subsets):
			# Calculate summation layer
			# summation_layer[j] = np.sum(
			# 	laplas(test_point, subsepip install adjustTextt[0], sigma)) / (subset[0].shape[0]*2*pow(sigma,d)*between/within)
			if tag==1:
				summation_layer[j] = np.sum(
				gas(test_point, subset[0], sigma)) / (subset[0].shape[0] *pow(2*np.pi, d/2)* pow(sigma,d))
			elif tag==2:
				summation_layer[j] = np.sum(
					mgas(test_point, subset[0], sigma)) / (subset[0].shape[0] * pow(2 * np.pi, d / 2) * pow(sigma, d))
			elif tag==3:
				summation_layer[j] = np.sum(
					cosdistance(test_point, subset[0], sigma)) / (subset[0].shape[0] * pow(2 * np.pi, d / 2) * pow(sigma, d))
			elif tag==4:
				summation_layer[j] = np.sum(
			elaplas(test_point, subset[0], sigma)) / (subset[0].shape[0]*2*pow(sigma,d)*between/within)
			elif tag==5:
				summation_layer[j] = np.sum(
					laplas(test_point, subset[0], sigma)) / (subset[0].shape[0] * 2 * pow(sigma, d) *between / within)
			elif tag==6:
				summation_layer[j] = np.sum(
					colaplas(test_point, subset[0], sigma)) / (subset[0].shape[0] * 2 * pow(sigma, d) * between / within)

		# print(summation_layer, np.argmax(summation_layer),data['y_test'][i])
		for  n in range(len(summation_layer)):
			# if n in data['y_test']:
			# 	print(np.unique(data['y_test']))
			summation_layer[n]=format(summation_layer[n],".3e")
			# else:
			# 	summation_layer[n] =0
		# ax.cla()  # 清除键
		# ax.set_yticklabels(nm, fontsize=12)
		# # plt.xlim(0, 14)
		# # ax.set_xlim([0, 14])
		# ax.set_yticks(nm)
		# ax.set_yticklabels(tlabel, minor=False)
		# rects = ax.barh(nm, label='GroundTruth:' + str(data['y_test'][i]) + "---predict---" + str(np.argmax(summation_layer)),
		# 			   height=0.1, width=summation_layer)
		# ax.set_title('GroundTruth:' + str(tlabel[data['y_test'][i]]) + "---predict---" + str(tlabel[np.argmax(summation_layer)]))
		# for rect in rects:
		# 	ax.text(rect.get_width(), rect.get_y(), rect.get_width(), ha='left', va='bottom')
		#
		# fig.lines = []
		# for joint_connection in SkeletonConnectionMap:
		# 	endpoint_x = [joints[joint_connection[0]][0], joints[joint_connection[1]][0]]
		# 	endpoint_y = [joints[joint_connection[0]][1], joints[joint_connection[1]][1]]
		# 	endpoint_z = [joints[joint_connection[0]][2], joints[joint_connection[1]][2]]
		# 	fig.plot(endpoint_x, endpoint_y, endpoint_z, c='r')
		# plt.pause(0.1)
		# The index having the largest value in the summation_layer is stored as the prediction
		predictions[i] = np.argmax(summation_layer)
		print([k for k,v in dic.items() if v==predictions[i]])
		if i==0:
			continue
		elif i==predictions.shape[0]-1:
			thread = threading.Thread(target=action_task, args=([k for k, v in dic.items() if v == predictions[i - 1]]))
			thread.start()
			time.sleep(5)
		else:
			if predictions[i]!=predictions[i-1]:
				thread=threading.Thread(target=action_task,args=([k for k,v in dic.items() if v==predictions[i-1]]))
				thread.start()
				time.sleep(5)
	return predictions


def print_metrics(y_test, predictions):
	predictions=predictions.astype(int)
	print('Confusion Matrix')
	print(y_test,predictions)
	# print(confusion_matrix(y_test, predictions))
	print('Accuracy: {}'.format(accuracy_score(y_test, predictions)))
	print('Precision: {}'.format(precision_score(y_test, predictions, average = 'macro')))
	print('Recall: {}'.format(recall_score(y_test, predictions, average = 'macro')))
	print('F1: {}'.format(f1_score(y_test, predictions, average='macro')))
if __name__ == '__main__':
	data = read_data.input(trainpath = 'labeled_data_train_all2.csv',testpath= 'testdata/A2s8c2-w.csv')
	precision=PNN(data, 0.01867524 , 3)
	print_metrics(data["y_test"],precision)
	#0.003603420.00360342
	#0.01266567
	#0.01866333
	# prediction1 = PNN(data, 0.00746044, 1)
	# prediction3 = PNN(data, 0.01867524, 3)
	# prediction4 = PNN(data, 0.00011233, 4)
	# prediction6 = PNN(data,0.00069621,6)
	# np.save("/home/hexin/桌面/new_warsadata/exampletest/groundtruth.npy",data['y_test'])
	# np.save("/home/hexin/桌面/new_warsadata/exampletest/E-G.npy", prediction1)
	# np.save("/home/hexin/桌面/new_warsadata/exampletest/ang-G.npy", prediction3)
	# np.save("/home/hexin/桌面/new_warsadata/exampletest/E-L.npy", prediction4)
	# np.save("/home/hexin/桌面/new_warsadata/exampletest/ang-L.npy", prediction6)
	# print(data['y_test'])
	# predictions=PNN(data,0.00069621,6)
	# print(data['y_test'].shape,predictions.shape)
	# print_metrics(data['y_test'], predictions)
	# print(accuracy_score(data['y_test'], predictions))
	# print(precision_score(data['y_test'], predictions, average = 'macro'))
	# print(recall_score(data['y_test'], predictions, average = 'macro'))
	# print(f1_score(data['y_test'], predictions, average='macro'))
	# ac1=[]
	# ac2=[]
	# ac3=[]
	# ac4 = []
	# ac5 = []
	# ac6 = []
	# ps1=[]
	# ps2 = []
	# ps3 = []
	# ps4 = []
	# ps5 = []
	# ps6 = []
	# rc1=[]
	# rc2 = []
	# rc3 = []
	# rc4 = []
	# rc5 = []
	# rc6 = []
	# F11=[]
	# F12 = []
	# F13 = []
	# F14 = []
	# F15 = []
	# F16 = []
	# sigmas=[0.00001,0.0001,0.001,0.01,0.1,1,10]
	# name=["0.00001","0.0001","0.001","0.01","0.1","1","10"]
	# rss1=(np.load("/home/hexin/桌面/new_warsadata/Euc-g-2.npy")*100).tolist()
	# rss2 = (np.load("/home/hexin/桌面/new_warsadata/Man-g-2.npy")*100).tolist()
	# rss3 = (np.load("/home/hexin/桌面/new_warsadata/cos-g-2.npy")*100).tolist()
	# rss4 = (np.load("/home/hexin/桌面/new_warsadata/Euc-l-2.npy")*100).tolist()
	# rss5 = (np.load("/home/hexin/桌面/new_warsadata/Man-l-2.npy")*100).tolist()
	# rss6 = (np.load("/home/hexin/桌面/new_warsadata/cos-l-2.npy")*100).tolist()
	# ac1=rss1[0]
	# ac2 = rss2[0]
	# ac3 = rss3[0]
	# ac4 = rss4[0]
	# ac5 = rss5[0]
	# ac6 = rss6[0]
	# ps1 = rss1[1]
	# ps2 = rss2[1]
	# ps3 = rss3[1]
	# ps4 = rss4[1]
	# ps5 = rss5[1]
	# ps6 = rss6[1]
	# rc1 = rss1[2]
	# rc2 = rss2[2]
	# rc3 = rss3[2]
	# rc4 = rss4[2]
	# rc5 = rss5[2]
	# rc6 = rss6[2]
	# F11 = rss1[3]
	# F12 = rss2[3]
	# F13 = rss3[3]
	# F14 = rss4[3]
	# F15 = rss5[3]
	# F16 = rss6[3]
	# # for sigma in sigmas:
	# 	# predictions1 = PNN(data,sigma,1)
	# 	# predictions2 = PNN(data, sigma, 2)
	# 	# predictions3 = PNN(data, sigma, 3)
	# 	# predictions4 = PNN(data, sigma, 4)
	# 	# predictions5 = PNN(data, sigma, 5)
	# 	# predictions6 = PNN(data, sigma, 6)
	# 	# print_metrics(data['y_test'], predictions)
	# 	# ac1.append(accuracy_score(data['y_test'], predictions1))
	# 	# ps1.append(precision_score(data['y_test'], predictions1, average = 'macro'))
	# 	# rc1.append(recall_score(data['y_test'], predictions1, average = 'macro'))
	# 	# F11.append(f1_score(data['y_test'], predictions1, average='macro'))
	# 	# ac2.append(accuracy_score(data['y_test'], predictions2))
	# 	# ps2.append(precision_score(data['y_test'], predictions2, average='macro'))
	# 	# rc2.append(recall_score(data['y_test'], predictions2, average='macro'))
	# 	# F12.append(f1_score(data['y_test'], predictions2, average='macro'))
	# 	# ac3.append(accuracy_score(data['y_test'], predictions3))
	# 	# ps3.append(precision_score(data['y_test'], predictions3, average='macro'))
	# 	# rc3.append(recall_score(data['y_test'], predictions3, average='macro'))
	# 	# F13.append(f1_score(data['y_test'], predictions3, average='macro'))
	# 	# ac4.append(accuracy_score(data['y_test'], predictions4))
	# 	# ps4.append(precision_score(data['y_test'], predictions4, average='macro'))
	# 	# rc4.append(recall_score(data['y_test'], predictions4, average='macro'))
	# 	# F14.append(f1_score(data['y_test'], predictions4, average='macro'))
	# 	# ac5.append(accuracy_score(data['y_test'], predictions5))
	# 	# ps5.append(precision_score(data['y_test'], predictions5, average='macro'))
	# 	# rc5.append(recall_score(data['y_test'], predictions5, average='macro'))
	# 	# F15.append(f1_score(data['y_test'], predictions5, average='macro'))
	# 	# ac6.append(accuracy_score(data['y_test'], predictions6))
	# 	# ps6.append(precision_score(data['y_test'], predictions6, average='macro'))
	# 	# rc6.append(recall_score(data['y_test'], predictions6, average='macro'))
	# 	# F16.append(f1_score(data['y_test'], predictions6, average='macro'))
	# ac1=[round(x,2) for x in  ac1]
	# ac2 = [round(x, 2) for x in ac2]
	# ac3 = [round(x, 2) for x in ac3]
	# ac4 = [round(x, 2) for x in ac4]
	# ac5 = [round(x, 2) for x in ac5]
	# ac6 = [round(x, 2) for x in ac6]
	# ps1 = [round(x, 2) for x in ps1]
	# ps2 = [round(x, 2) for x in ps2]
	# ps3 = [round(x, 2) for x in ps3]
	# ps4 = [round(x, 2) for x in ps4]
	# ps5 = [round(x, 2) for x in ps5]
	# ps6 = [round(x, 2) for x in ps6]
	# rc1 = [round(x, 2) for x in rc1]
	# rc2 = [round(x, 2) for x in rc2]
	# rc3 = [round(x, 2) for x in rc3]
	# rc4 = [round(x, 2) for x in rc4]
	# rc5 = [round(x, 2) for x in rc5]
	# rc6 = [round(x, 2) for x in rc6]
	# F11 = [round(x, 2) for x in F11]
	# F12 = [round(x, 2) for x in F12]
	# F13 = [round(x, 2) for x in F13]
	# F14 = [round(x, 2) for x in F14]
	# F15 = [round(x, 2) for x in F15]
	# F16 = [round(x, 2) for x in F16]
	# #
	# #
	# #
	# ax = plt.subplot(2, 2, 1)
	# ax.plot(name,ac1,label="Euc-g",color="r")
	# ax.plot(name, ac2, label="Man-g", color="g")
	# ax.plot(name, ac3, label="cos-g", color="b")
	# ax.plot(name, ac4, label="Euc-l", color="y")
	# ax.plot(name, ac5, label="Man-l", color="orange")
	# ax.plot(name, ac6, label="cos-l", color="purple")
	# t1 = [plt.text(a, b, b, ha="center", va="bottom") for a, b in zip(name, ac1)]
	# t2 = [plt.text(a, b, b, ha="center", va="bottom") for a, b in zip(name, ac2)]
	# t3 = [plt.text(a, b, b, ha="center", va="bottom") for a, b in zip(name, ac3)]
	# # adjust_text(t1, )
	# # adjust_text(t2, )
	# # adjust_text(t3, )
	# t4 = [plt.text(a, b, b, ha="center", va="bottom") for a, b in zip(name, ac4)]
	# t5 = [plt.text(a, b, b, ha="center", va="bottom") for a, b in zip(name, ac5)]
	# t6 = [plt.text(a, b, b, ha="center", va="bottom") for a, b in zip(name, ac6)]
	# # adjust_text(t4, )
	# # adjust_text(t5, )
	# # adjust_text(t6, )
	# plt.legend()
	# plt.ylabel("accuracy")
	# plt.xlabel("sigma")
	# ax2= plt.subplot(2,2, 2)
	# plt.ylabel("pricision")
	# plt.xlabel("sigma")
	# ax2.plot(name, ps1, label="Euc-g", color="r")
	# ax2.plot(name, ps2, label="Man-g", color="g")
	# ax2.plot(name, ps3, label="cos-g", color="b")
	# ax2.plot(name, ps4, label="Euc-l", color="y")
	# ax2.plot(name, ps5, label="Man-l", color="orange")
	# ax2.plot(name, ps6, label="cos-l", color="purple")
	# t1 = [plt.text(a, b, b, ha="center", va="bottom") for a, b in zip(name, ps1)]
	# t2 = [plt.text(a, b, b, ha="center", va="bottom") for a, b in zip(name, ps2)]
	# t3 = [plt.text(a, b, b, ha="center", va="bottom") for a, b in zip(name, ps3)]
	# # adjust_text(t1, )
	# # adjust_text(t2, )
	# # adjust_text(t3, )
	# t4 = [plt.text(a, b, b, ha="center", va="bottom") for a, b in zip(name, ps4)]
	# t5 = [plt.text(a, b, b, ha="center", va="bottom") for a, b in zip(name, ps5)]
	# t6 = [plt.text(a, b, b, ha="center", va="bottom") for a, b in zip(name, ps6)]
	# # adjust_text(t4, )
	# # adjust_text(t5, )
	# # adjust_text(t6, )
	# plt.legend()
	# ax3 = plt.subplot(2, 2, 3)
	# plt.ylabel("recall")
	# plt.xlabel("sigma")
	# ax3.plot(name, rc1, label="Euc-g", color="r")
	# ax3.plot(name, rc2, label="Man-g", color="g")
	# ax3.plot(name, rc3, label="cos-g", color="b")
	# ax3.plot(name, rc4, label="Euc-l", color="y")
	# ax3.plot(name, rc5, label="Man-l", color="orange")
	# ax3.plot(name, rc6, label="cos-l", color="purple")
	# t1 = [plt.text(a, b, b, ha="center", va="bottom") for a, b in zip(name, rc1)]
	# t2 = [plt.text(a, b, b, ha="center", va="bottom") for a, b in zip(name, rc2)]
	# t3 = [plt.text(a, b, b, ha="center", va="bottom") for a, b in zip(name, rc3)]
	# # adjust_text(t1, )
	# # adjust_text(t2, )
	# # adjust_text(t3, )
	# t4=[plt.text(a,b,b,ha="center",va="bottom") for a,b in zip(name,rc4)]
	# t5 = [plt.text(a, b, b, ha="center", va="bottom") for a, b in zip(name, rc5)]
	# t6 = [plt.text(a, b, b, ha="center", va="bottom") for a, b in zip(name, rc6)]
	# # adjust_text(t4, )
	# # adjust_text(t5, )
	# # adjust_text(t6, )
	# plt.legend()
	# ax4 = plt.subplot(2, 2, 4)
	# plt.ylabel("f1-score")
	# plt.xlabel("sigma")
	# ax4.plot(name, F11, label="Euc-g", color="r")
	# ax4.plot(name, F12, label="Man-g", color="g")
	# ax4.plot(name, F13, label="cos-g", color="b")
	# ax4.plot(name, F14, label="Euc-l", color="y")
	# ax4.plot(name, F15, label="Man-l", color="orange")
	# ax4.plot(name, F16, label="cos-l", color="purple")
	# t1 = [plt.text(a, b, b, ha="center", va="bottom") for a, b in zip(name, F11)]
	# t2 = [plt.text(a, b, b, ha="center", va="bottom") for a, b in zip(name, F12)]
	# t3 = [plt.text(a, b, b, ha="center", va="bottom") for a, b in zip(name, F13)]
	# # adjust_text(t1, )
	# # adjust_text(t2, )
	# # adjust_text(t3, )
	# t4 = [plt.text(a, b, b, ha="center", va="bottom") for a, b in zip(name, F14)]
	# t5 = [plt.text(a, b, b, ha="center", va="bottom") for a, b in zip(name, F15)]
	# t6 = [plt.text(a, b, b, ha="center", va="bottom") for a, b in zip(name, F16)]
	# # adjust_text(t4, )
	# # adjust_text(t5, )
	# # adjust_text(t6, )
	# plt.legend()
	# # sav=np.array([ac1,ps1,rc1,F11])
	# # np.save("/home/hexin/桌面/new_warsadata/Euc-g-3.npy",sav)
	# # sav=np.array([ac2,ps2,rc2,F12])
	# # np.save("/home/hexin/桌面/new_warsadata/Man-g-3.npy",sav)
	#
	# # sav=np.array([ac3,ps3,rc3,F13])
	# # np.save("/home/hexin/桌面/new_warsadata/cos-g-2.npy",sav)
	# # sav=np.array([ac4,ps4,rc4,F14])
	# # np.save("/home/hexin/桌面/new_warsadata/Euc-l-2.npy",sav)
	# # sav = np.array([ac5, ps5, rc5, F15])
	# # np.save("/home/hexin/桌面/new_warsadata/Man-l-2.npy", sav)
	# # sav = np.array([ac6, ps6, rc6, F16])
	# # np.save("/home/hexin/桌面/new_warsadata/cos-l-2.npy", sav)
	# plt.show()
	# print("d")