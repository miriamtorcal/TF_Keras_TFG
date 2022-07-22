import itertools
from operator import index
from absl import app, flags
from absl.flags import FLAGS
import cv2
import pandas as pd
import os
import warnings

# Quitar warnings
warnings.simplefilter(action='ignore')   

flags.DEFINE_string('csv', './result.csv', 'path to input csv')
flags.DEFINE_string('positions', './original_data/positions', 'path to input original positions')
flags.DEFINE_string('classes', './original_data/classes', 'path to input classes in positions')
flags.DEFINE_string('img_video', './data/video', 'path to')
flags.DEFINE_boolean('accuracy', False, 'accuracy per class')


def main(_argv):
    dataCsv = {}
    classesData = []
    csvPath = FLAGS.csv
    posPath = FLAGS.positions
    classesPath = FLAGS.classes
    img_data = cv2.imread(FLAGS.img_video)
    height, weight, _ = img_data.shape

    # Read csv file and get the data in a df
    csvDf = pd.read_csv(csvPath)

    for i in csvDf.index:
        className = csvDf['TypeObject'][i]
        dataCsv.setdefault(className, [])
        pos = csvDf['Positions'][i]
        
        pos = pos.replace('[', '')
        pos = pos.replace(']', '')
        pos = pos.replace('\'', '')
        pos = pos.split('),')

        for k in range(len(pos)):
            p = pos[k].replace('(', '')
            p = p.replace(')', '')
            dataCsv[className].append(p)

    # Read file class and save in a list
    classesFich = open(classesPath, "r")
    classData = classesFich.read()
    text = ''
    for i in classData:
        text += i
        if (i == '\n'):
            text = text.replace('\n', '')
            classesData.append(text)
            text = ''

    #Read all files with positions per image
    content = os.listdir(posPath)
    headers = ['class', 'xMin', 'yMin', 'xMax', 'yMax']
    dataPos = pd.DataFrame()
    
    for i in headers:
        dataPos[i] = ""

    for file in content:
        if os.path.isfile(os.path.join(posPath, file)) and file.endswith(".txt"):
            posDf = pd.read_table(posPath + file, sep=" ", names=headers)
            dataPos = pd.concat([dataPos, posDf])

    # print(dataPos.to_numpy().tolist())

    for l in dataPos.index:
        # for k in range(len(classesData)):
            # if(dataPos["class"][l] == k):
            #     dataPos["class"].loc[l] = classesData[k]
        dataPos["xMin"][l] = int(dataPos["xMin"][l] * height)
        dataPos["yMin"][l] = int(dataPos["yMin"][l] * weight)
        dataPos["xMax"][l] = int(dataPos["xMax"][l] * height)
        dataPos["yMax"][l] = int(dataPos["yMax"][l] * weight)

    dataPos["xMin"] = dataPos["xMin"].astype(int)
    dataPos["yMin"] = dataPos["yMin"].astype(int)
    dataPos["xMax"] = dataPos["xMax"].astype(int)
    dataPos["yMax"] = dataPos["yMax"].astype(int)


    dataPos = dataPos[['xMin','yMin','xMax','yMax','class']]

    listDP = dataPos.to_numpy().tolist()
    # listDP = list(itertools.chain(*listDP))
    # strDP = " ".join([str(_) for _ in listDP])
    # print(strDP)

    strPos = ""
    for _ in range(len(listDP)):
        # print(listDP[_])
        strDP = ",".join([str(_) for _ in listDP[_]])
        if (_ != len(listDP) - 1):
            strDP = strDP + ' '
        strPos += strDP

    with open('./data/dataset/cars.txt', 'w') as f:
        f.write(FLAGS.img_video + ' ' + strPos)
    
    #Calculate accuracy
    listDf = separateDfByClass(csvDf, classesData)
    dfSP = pd.DataFrame(columns=['class', 'pos'])
    for k in range(len(listDf)):
        pos = listDf[k]['Positions']
        for i in pos:
            i = i.replace("[", "")
            i = i.replace("]", "")
            listPos = list(i.split(", ("))
            for _ in listPos:
                if (_.find('(') != -1):
                    _ =_.replace("(", "")
                _ = _.replace(")", "")
                # list_ = list(_.split(", "))
                # list_[0] = float(list_[0]) / height
                # list_[1] = float(list_[1]) / weight
                # list_[2] = float(list_[2]) / height
                # list_[3] = float(list_[3]) / weight
                # # print(list_)
                # _ = " ".join([str(l) for l in list_])
                className = list(listDf[k]['TypeObject'])[0]
                row = {'class':className, 'pos':_}
                dfSP = dfSP.append(row, ignore_index=True)
    print(dfSP)

def separateDfByClass(df: pd.DataFrame, classesData: list) -> list:
    numberDf = len(classesData)
    listDf = []

    for i in range(0, numberDf):
        groups = df.groupby(df.TypeObject, group_keys=False)
        newdf = groups.get_group(classesData[i])
        listDf.append(newdf)

    return listDf

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass