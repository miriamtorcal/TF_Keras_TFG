La carpeta yolov4 y su contenido deberá de encontrarse en Drive, ya que su contenido es necesario para el entrenamiento.

```
├── yolov4
│   ├── generate_test.py    # Genera el conjunto de test
│   ├── generate_train.py   # Genera el conjunto de train
│   ├── obj.data            # Contiene información sobre el mddelo que se va entrenar (revisar previamente)
│   ├── obj.names           # Nombres de las clases que detectara el modelo
│   ├── yolov4-obj.cfg      # Fichero configuración yolov4
```

[![Open Folder](https://img.shields.io/badge/Open%20Folder-yolov4-%232bb5b5?logo=yolo)](https://github.com/mtc1003/TF_Keras_TFG/blob/main/codigo/yolov4ToTensorFlow/train/yolov4)

train.ipynb es el fichero a utilizar para el entrenamiento del modelo 

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mtc1003/TF_Keras_TFG/blob/main/codigo/yolov4ToTensorFlow/train/train.ipynb)
