from operator import index
from turtle import width
from absl import app, flags
from absl.flags import FLAGS
import cv2
import pandas as pd
import os
import warnings

# Quitar warnings
warnings.simplefilter(action='ignore')   

flags.DEFINE_string('positions', './original_data/positions', 'path to input original positions')

def main(_argv):
    posPath = FLAGS.positions
    listDf = []
    files = []

    #Read all files with positions per image
    content = os.listdir(posPath)
    headers = ['class', 'x', 'y', 'w', 'h']
    dataPos = pd.DataFrame()
    
    for i in headers:
        dataPos[i] = ""
        dataPos[i] = dataPos[i].astype(float)

    for file in content:
        if os.path.isfile(os.path.join(posPath, file)) and file.endswith(".txt"):
            posDf = pd.read_table(posPath + file, sep=" ", names=headers)
            file = file.replace('.txt', '.jpg')
            posDf['file'] = posPath + file
            # print(posDf)
            files.append(posPath + file)
            dataPos = pd.concat([dataPos, posDf])

    dataPos = dataPos.reset_index(drop=True)

    for l in dataPos.index:
        f = dataPos["file"][l]
        img = cv2.imread(f)
        height, width, _ = img.shape
        
        x = dataPos["x"].iloc[l]
        y = dataPos["y"].iloc[l]
        w = dataPos["w"].iloc[l]
        h = dataPos["h"].iloc[l]
        
        dataPos["x"].iloc[l] = int((x - w / 2) * width)
        dataPos["y"].iloc[l] = int((y - h / 2) * height)
        dataPos["w"].iloc[l] = int((x + w / 2) * width)
        dataPos["h"].iloc[l] = int((y + h / 2) * height)

    dataPos["x"] = dataPos["x"].astype(int)
    dataPos["y"] = dataPos["y"].astype(int)
    dataPos["w"] = dataPos["w"].astype(int)
    dataPos["h"] = dataPos["h"].astype(int)
    dataPos['class'] = dataPos["class"].astype(int)

    # print(dataPos)

    dataPos = dataPos[['file','x','y','w','h','class']]


    groups = dataPos.groupby(dataPos.file, group_keys=False)
    for g in range(len(groups)):
        newdf = groups.get_group(files[g])
        listDf.append(newdf)

    strPos = []
    for l in range(len(listDf)):
        df = listDf[l]
        df = df.reset_index(drop=True)

        df = df.drop(['file'], axis=1)

        listDF = df.to_numpy().tolist()
        strLine = ""
        for _ in range(len(listDF)):
            strDF = ",".join([str(_) for _ in listDF[_]])
            strDF = strDF.replace('.jpg,', '.jpg ')
            if (_ != len(listDF) - 1):
                strDF = strDF + ' '
            strLine += strDF
        strPos.append(strLine)

    with open('./data/dataset/license_plate.txt', 'w') as f:
        for w in range(len(strPos)):
            f.write(files[w] + ' ' + strPos[w])
            f.write('\n')

    print("File generate suscessfully!!!")

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass