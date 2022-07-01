from absl import app, flags
from absl.flags import FLAGS
import cv2
import pandas as pd
import os
import warnings

# Quitar warnings
warnings.simplefilter(action='ignore')   

flags.DEFINE_string('positions', './original_data/positions', 'path to input original positions')
flags.DEFINE_string('name_txt_export', 'head.txt', 'name of the txt with all positions')

def main(_argv):
    pos_path = FLAGS.positions
    list_df = []
    files = []

    #Read all files with positions per image
    content = os.listdir(pos_path)
    headers = ['class', 'x', 'y', 'w', 'h']
    data_pos = pd.DataFrame()
    
    for i in headers:
        data_pos[i] = ""
        data_pos[i] = data_pos[i].astype(float)

    for file in content:
        if os.path.isfile(os.path.join(pos_path, file)) and file.endswith(".txt"):
            pos_df = pd.read_table(pos_path + file, sep=" ", names=headers)
            if pos_df.empty:
                continue
            file = file.replace('.txt', '.png')
            pos_df['file'] = pos_path + file 
            files.append(pos_path + file)
            data_pos = pd.concat([data_pos, pos_df])

    data_pos = data_pos.reset_index(drop=True)

    for l in data_pos.index:
        f = data_pos["file"][l]
        img = cv2.imread(f)
        height, width, _ = img.shape
        
        x = data_pos["x"].iloc[l]
        y = data_pos["y"].iloc[l]
        w = data_pos["w"].iloc[l]
        h = data_pos["h"].iloc[l]
        
        data_pos["x"].iloc[l] = int((x - w / 2) * width)
        data_pos["y"].iloc[l] = int((y - h / 2) * height)
        data_pos["w"].iloc[l] = int((x + w / 2) * width)
        data_pos["h"].iloc[l] = int((y + h / 2) * height)

    data_pos["x"] = data_pos["x"].astype(int)
    data_pos["y"] = data_pos["y"].astype(int)
    data_pos["w"] = data_pos["w"].astype(int)
    data_pos["h"] = data_pos["h"].astype(int)
    data_pos['class'] = data_pos["class"].astype(int)

    data_pos = data_pos[['file','x','y','w','h','class']]

    groups = data_pos.groupby(data_pos.file, group_keys=False)
    for g in range(len(groups)):
        newdf = groups.get_group(files[g])
        list_df.append(newdf)

    str_pos = []
    for l in range(len(list_df)):
        df = list_df[l]
        df = df.reset_index(drop=True)

        df = df.drop(['file'], axis=1)

        list_list_df = df.to_numpy().tolist()
        str_line = ""
        for _ in range(len(list_list_df)):
            str_df = ",".join([str(_) for _ in list_list_df[_]])
            str_df = str_df.replace('.png,', '.png ')
            if (_ != len(list_list_df) - 1):
                str_df = str_df + ' '
            str_line += str_df
        str_pos.append(str_line)

    with open('./data/dataset/' + FLAGS.name_txt_export, 'w') as f:
        for w in range(len(str_pos)):
            f.write(files[w] + ' ' + str_pos[w])
            f.write('\n')

    print("File generate suscessfully!!!")

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass