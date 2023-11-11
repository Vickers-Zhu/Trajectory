# coding: utf-8

import random
import math
import numpy as np
import matplotlib.pyplot as plt

file=open("20201206-S6F1820#1S20.trj")
lines=file.readlines()
file.close()

target=39

xyz = [ ]
frames = [ ]

x0=0
y0=0
z0=0
vx0=0
vy0=0
vz0=0

dt = 1.0/60

cnt=0
for line in lines:
    cols=line.split()
    if len(cols)<6:
        continue
    tid=int(cols[0])
    if tid!=target:
        continue

    if cnt==0:
        x0 = float(cols[6])
        y0 = float(cols[7])
        z0 = float(cols[8])
    elif cnt==1:
        vx0 = (float(cols[6]) - x0)/dt
        vy0 = (float(cols[7]) - y0)/dt
        vz0 = (float(cols[8]) - z0)/dt
        # print(x0,y0,z0,vx0,vy0,vz0)

    t=int(cols[1])
    x=float(cols[3])
    y=float(cols[4])
    z=float(cols[5])
    frames.append(t)
    xyz.append([x,y,z])
    cnt+=1

xyz = np.array(xyz)    

print(cnt)    

F = np.array([[1, 0, 0, dt,  0,  0,  0,  0,  0],
              [0, 1, 0,  0, dt,  0,  0,  0,  0],
              [0, 0, 1,  0,  0, dt,  0,  0,  0],
              [0, 0, 0,  1,  0,  0, dt,  0,  0],
              [0, 0, 0,  0,  1,  0,  0, dt,  0],
              [0, 0, 0,  0,  0,  1,  0,  0, dt],
              [0, 0, 0,  0,  0,  0,  1,  0,  0],
              [0, 0, 0,  0,  0,  0,  0,  1,  0],
              [0, 0, 0,  0,  0,  0,  0,  0,  1]])

G = np.array([[0,  0,  0],
              [0,  0,  0],
              [0,  0,  0],
              [0,  0,  0],
              [0,  0,  0],
              [0,  0,  0],
              [math.sqrt(dt),  0,  0],
              [0,  math.sqrt(dt),  0],
              [0,  0,  math.sqrt(dt)]])

H = np.array([[1,0,0,0,0,0,0,0,0],
              [0,1,0,0,0,0,0,0,0],
              [0,0,1,0,0,0,0,0,0]])

X = np.zeros((9,1))
XT = np.zeros((9,1))

S = np.zeros((9,9))

Q = np.array([[0.5,   0,   0],
              [  0, 0.5,   0],
              [  0,   0, 0.5],              
              ])

R = np.array([[0.3,   0,   0],
              [  0, 0.3,   0],
              [  0,   0, 0.3],              
              ])



XT[0,0]=x0
XT[1,0]=y0
XT[2,0]=z0
XT[3,0]=vx0
XT[4,0]=vy0
XT[5,0]=vz0
XT[6,0]=0
XT[7,0]=0
XT[8,0]=0

S=np.eye(9)

TIM=[]
PX=[]
PY=[]
PZ=[]
VX=[]
VY=[]
VZ=[]
VEL=[]
AX=[]
AY=[]
AZ=[]
ACC=[]
AXERR=[]

t=0
for i in range(len(frames)):

    if i>0 and frames[i] - frames[i-1]>1:
        print('GAP',i,frames[i])
    
    D = np.linalg.pinv(H.dot(S.dot(H.T)) + R)
    K = S.dot(H.T).dot(D)

    # observation
    Z = np.array([xyz[i,:]]).T

    X2 = XT + K.dot(Z - H.dot(XT))
    S2 = (np.eye(9) - K.dot(H)).dot(S)

    TIM.append(frames[i])
    PX.append(XT[0,0])
    PY.append(XT[1,0])
    PZ.append(XT[2,0])
    VX.append(XT[3,0])
    VY.append(XT[4,0])
    VZ.append(XT[5,0])
    AX.append(XT[6,0])
    AY.append(XT[7,0])
    AZ.append(XT[8,0])
    AXERR.append(math.sqrt(S[6,6]))
    ACC.append(math.sqrt(XT[6,0]**2 + XT[7,0]**2+XT[8,0]**2))
    VEL.append(math.sqrt(XT[3,0]**2 + XT[4,0]**2+XT[5,0]**2))

    XT = F.dot(X2)
    S = F.dot(S2.dot(F.T)) + G.dot(Q.dot(G.T))

    t = t + dt        



with open('acc_data/' + str(target)+'.txt', 'w') as f:
    for i in range(len(frames)):
        print(frames[i],AX[i],AY[i],AZ[i],file=f)

    
plt.plot(TIM,AX, color='blue', linewidth=1.0, label='AX')
plt.plot(TIM,AY, color='green', linewidth=1.0, label='AY')
plt.plot(TIM,AZ, color='red', linewidth=1.0, label='AZ')
# plt.plot(TIM,ACC, color='red', linewidth=1.0, label='ACC')
# plt.plot(TIM,VX, color='blue', linewidth=1.0, label='VX')
# plt.plot(TIM,VY, color='green', linewidth=1.0, label='VY')
# plt.plot(TIM,VZ, color='red', linewidth=1.0, label='VZ')
# plt.plot(TIM,VEL, color='red', linewidth=1.0, label='VELOC')
# plt.errorbar(TIM,AX, yerr=AXERR, capsize=1, fmt='.', color='red', ecolor='orange',label='AX Estimate')
# plt.plot(TIM,xyz[:,1], color='blue', linewidth=1.0, label='obs')
# plt.plot(TIM,PY, color='green', linewidth=1.0, label='Y')
plt.title('trajectory #'+str(target))
plt.xlabel('T')
plt.ylabel('X')
plt.grid(True)
plt.legend()
plt.xlim(1800,2700)
plt.show()
