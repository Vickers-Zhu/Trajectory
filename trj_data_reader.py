import TrjSample

def read_trj_data(filename):
    with open(filename,mode='r',encoding='utf-8') as file:
        lines = file.readlines()
    data=[ ]
    cnt=0
    trj_prev=-1
    for line in lines:
        cols = line.split()
        if len(cols)>1: # skip empty line
            print(len(cols))
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
                trajectory.append([frame,xpos,ypos,zpos,xvel,yvel,zvel])
    return data

from TrjSample import TrjSample


def read_trj_fulldata(filename):
    trj_samples = []  # Initialize an empty list to store TrjSample objects
    # Read the data from the file and create TrjSample objects
    with open(filename, "r") as file:
        for line in file:
            values = line.split()
            if len(values) != 19:
                print('continue')
                continue
            trj_sample = TrjSample.TrjSample(
                int(values[0]),
                int(values[1]),
                int(values[2]),
                float(values[3]),
                float(values[4]),
                float(values[5]),
                float(values[6]),
                float(values[7]),
                float(values[8]),
                float(values[9]),
                float(values[10]),
                float(values[11]),
                float(values[12]),
                float(values[13]),
                float(values[14]),
                int(values[15]),
                float(values[16]),
                float(values[17]),
                float(values[18])
            )
            trj_samples.append(trj_sample)
    
    return trj_samples
