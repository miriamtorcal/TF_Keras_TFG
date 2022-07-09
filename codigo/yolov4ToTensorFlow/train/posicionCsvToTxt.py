import pandas as pd
import os

working_dir = 'D:\Proyectos\heads_dataset\Testing_set_Blind\Label'


df = pd.read_csv("D:\Proyectos\heads_dataset\Testing_set_Blind\_labels_TestingSet_Blind.csv", names=['Img', 'Pos'])

# df["Img"]



for x in range(len(df['Img'])):
    img = df["Img"].iloc[x].replace('.png', "")
    img = img.replace("'", '')

    if len(df["Pos"].iloc[x]) != 2:
        pos = df["Pos"].iloc[x].replace('[', "")
        pos = pos.replace(']', "")

        pos = pos.split(';')

        file = open(working_dir + '\\' + img + ".txt", "w")
        for k in pos:
            if pos != [' ']:
                if k != '':
                    coords = list(map(str,k.split(',')))
                    x_min = int(coords[0].strip()[0:3])
                    y_min = int(coords[0].strip()[5:8])
                    width = int(coords[0].strip()[10:13])
                    height = int(coords[0].strip()[15:18])
                    x_max = x_min + width        
                    y_max = y_min + height
                    coords2 = [x_min, y_min, x_max, y_max]
                    # print(coords2)
                    coords2 = str(coords2).replace('[','').replace(']','').replace(' ','').replace(',', ' ')
                    file.write("Head " + coords2 + "\n")
        else:
         file = open(working_dir + '\\' + img + ".txt", "w")
