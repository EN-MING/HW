import pickle
with open ("D:\\test\\MLGame-master1\\games\\arkanoid\\log\\2019-10-29_10-04-08.pickle" , "rb") as f1:
    date_list_1 = pickle.load(f1)
with open ("D:\\test\\MLGame-master1\\games\\arkanoid\\log\\2019-10-29_10-06-07.pickle" , "rb") as f2:
    date_list_2 = pickle.load(f2)
with open ("D:\\test\\MLGame-master1\\games\\arkanoid\\log\\2019-10-29_10-05-07.pickle" , "rb") as f3:
    date_list_3 = pickle.load(f3)
with open ("D:\\test\\MLGame-master1\\games\\arkanoid\\log\\2019-10-29_10-07-06.pickle" , "rb") as f4:
    date_list_4 = pickle.load(f4)
with open ("D:\\test\\MLGame-master1\\games\\arkanoid\\log\\2019-10-29_10-08-05.pickle" , "rb") as f5:
    date_list_5 = pickle.load(f5)
with open ("D:\\test\\MLGame-master1\\games\\arkanoid\\log\\2019-10-29_10-09-05.pickle" , "rb") as f6:
    date_list_6 = pickle.load(f6)
with open ("D:\\test\\MLGame-master1\\games\\arkanoid\\log\\2019-10-29_10-10-04.pickle" , "rb") as f7:
    date_list_7 = pickle.load(f7)
with open ("D:\\test\\MLGame-master1\\games\\arkanoid\\log\\2019-10-29_10-11-03.pickle" , "rb") as f8:
    date_list_8 = pickle.load(f8)
with open ("D:\\test\\MLGame-master1\\games\\arkanoid\\log\\2019-10-29_10-12-02.pickle" , "rb") as f9:
    date_list_9 = pickle.load(f9)
Frame = [ ]
Status = [ ]
Ballposition = [ ]
Platformposition = [ ]
Bricks = [ ]
for i in range (0,len(date_list_1)):
    Frame.append(date_list_1[i].frame)
    Status.append(date_list_1[i].status)
    Ballposition.append(date_list_1[i].ball)
    Platformposition.append(date_list_1[i].platform)
    Bricks.append(date_list_1[i].bricks)
for i in range (0,len(date_list_2)):
    Frame.append(date_list_2[i].frame)
    Status.append(date_list_2[i].status)
    Ballposition.append(date_list_2[i].ball)
    Platformposition.append(date_list_2[i].platform)
    Bricks.append(date_list_2[i].bricks)
for i in range (0,len(date_list_3)):
    Frame.append(date_list_3[i].frame)
    Status.append(date_list_3[i].status)
    Ballposition.append(date_list_3[i].ball)
    Platformposition.append(date_list_3[i].platform)
    Bricks.append(date_list_3[i].bricks)
for i in range (0,len(date_list_4)):
    Frame.append(date_list_4[i].frame)
    Status.append(date_list_4[i].status)
    Ballposition.append(date_list_4[i].ball)
    Platformposition.append(date_list_4[i].platform)
    Bricks.append(date_list_4[i].bricks)
for i in range (0,len(date_list_5)):
    Frame.append(date_list_5[i].frame)
    Status.append(date_list_5[i].status)
    Ballposition.append(date_list_5[i].ball)
    Platformposition.append(date_list_5[i].platform)
    Bricks.append(date_list_5[i].bricks)
for i in range (0,len(date_list_6)):
    Frame.append(date_list_6[i].frame)
    Status.append(date_list_6[i].status)
    Ballposition.append(date_list_6[i].ball)
    Platformposition.append(date_list_6[i].platform)
    Bricks.append(date_list_6[i].bricks)
for i in range (0,len(date_list_7)):
    Frame.append(date_list_7[i].frame)
    Status.append(date_list_7[i].status)
    Ballposition.append(date_list_7[i].ball)
    Platformposition.append(date_list_7[i].platform)
    Bricks.append(date_list_7[i].bricks)
for i in range (0,len(date_list_8)):
    Frame.append(date_list_8[i].frame)
    Status.append(date_list_8[i].status)
    Ballposition.append(date_list_8[i].ball)
    Platformposition.append(date_list_8[i].platform)
    Bricks.append(date_list_8[i].bricks)
for i in range (0,len(date_list_9)):
    Frame.append(date_list_9[i].frame)
    Status.append(date_list_9[i].status)
    Ballposition.append(date_list_9[i].ball)
    Platformposition.append(date_list_9[i].platform)
    Bricks.append(date_list_9[i].bricks)


#----------------------------------------------------------------------------------------------------------------------------
import numpy as np
PlatX = np.array(Platformposition)[:,0][:,np.newaxis]
PlatX_next = PlatX[1:,:]
instruct = (PlatX_next - PlatX[0:len(PlatX_next),0][:, np.newaxis])/5

BallX = np.array(Ballposition)[:,0][:,np.newaxis]
BallX_next = BallX[1:,:]
vx = (BallX_next - BallX[0:len(BallX_next),0][:,np.newaxis])

BallY = np.array(Ballposition)[:,1][:,np.newaxis]
BallY_next = BallY[1:,:]
vy = (BallY_next - BallY[0:len(BallY_next),0][:,np.newaxis])

Ballarray = np.array(Ballposition[:-1])
x = np.hstack((Ballarray , PlatX[0:-1,0][:,np.newaxis],vx,vy))

y = instruct

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=25,random_state=999)

#----------------------------------------------------------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(x_train,y_train)

yknn_bef_scaler = knn.predict(x_test)
acc_knn_bef_scaler = accuracy_score(yknn_bef_scaler, y_test)


#----------------------------------------------------------------------------------------------------------------------------
'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_stdnorm = scaler.transform(x_train)
knn.fit(x_train_stdnorm ,y_train)
x_test_stdnorm = scaler.transform(x_test)
yknn_aft_scaler = knn.predict(x_test_stdnorm)
acc_knn_aft_scaler = accuracy_score(yknn_aft_scaler,y_test)
'''

#----------------------------------------------------------------------------------------------------------------------------
filename = "D:\\test\\MLGame-master1\\knn_example1.sav"
pickle.dump(knn , open(filename , 'wb'))