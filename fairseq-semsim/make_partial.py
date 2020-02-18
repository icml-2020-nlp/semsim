import sys

if len(sys.argv)<2 or sys.argv[1]==None:
    device = 4
else:
    try:
        device = int(sys.argv[1])
    except :
        print("The first argument, which means the number of arguments, should be integer")

# devide datasets
foutList = [open('cnn_dm/partial/test.source.part%d'%idx, 'w') for idx in range(0, device)]
with open('cnn_dm/test.source') as source:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        slines.append(sline.strip())

oneDevSize = int(len(slines)/device) +1
partSlines=[slines[start:start+oneDevSize] for start in range(0, len(slines), oneDevSize)]

numChk=0
for ele in partSlines:
    numChk+=len(ele)
assert numChk==len(slines)

for fileP, mini_slines in zip(foutList, partSlines):
    for sline in mini_slines:
        fileP.write('%s\n' %sline)
    fileP.close()


import numpy as np
np.random.seed(0)


with open('cnn_dm/test.target') as target:
    tline = target.readline().strip()
    tlines = [tline]
    for tline in target:
        tlines.append(tline.strip())

samDict = dict()
for idx, (sele,tele) in enumerate(zip(slines, tlines)):
    samDict[idx] = (sele,tele)

with open('cnn_dm/partial/test.source.sample-1000', 'w') as samout, open('cnn_dm/partial/test.target.sample-1000', 'w') as tarout:
    pickNumList = np.random.permutation(len(slines))[:1000]
    for pick in pickNumList:
        samout.write('%s\n' %samDict[pick][0])
        tarout.write('%s\n' %samDict[pick][1])

