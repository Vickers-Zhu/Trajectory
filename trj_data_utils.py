# import TrjSample
import matplotlib.pyplot as plt

def read_trj_data(filename):
    with open(filename,mode='r',encoding='utf-8') as file:
        lines = file.readlines()
    data=[ ]
    cnt=0
    trj_prev=-1
    for line in lines:
        cols = line.split()
        if len(cols)>1: # skip empty line
            trj_id = int(cols[0])
            frame = int(cols[1])
            xpos = float(cols[6])  # depth
            ypos = float(cols[7])
            zpos = float(cols[8])  # height
            xvel = float(cols[12])
            yvel = float(cols[13])
            zvel = float(cols[14])
            n = int(cols[15]) # length of sequence

            if trj_prev != trj_id:
                if trj_prev!=-1: # very first one
                    data.append(trajectory)
                trajectory = []
                trj_prev = trj_id
                cnt += 1
            else:
                trajectory.append([trj_id,frame,xpos,ypos,zpos,xvel,yvel,zvel])
    return data
