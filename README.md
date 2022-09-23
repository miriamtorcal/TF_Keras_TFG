<h4 align="center">Aplicaciones de Visión Artificial en Dispositivos de Edge Computing.</h4>
<div align="center">
  <img alt="SonarCloud Quality Gate" src="https://sonarcloud.io/api/project_badges/measure?project=mtc1003_TF_Keras_TFG&metric=alert_status">
  <img alt="SonarCloud Security" src="https://sonarcloud.io/api/project_badges/measure?project=mtc1003_TF_Keras_TFG&metric=security_rating">
  <img alt="SonarCloud Sqale" src="https://sonarcloud.io/api/project_badges/measure?project=mtc1003_TF_Keras_TFG&metric=sqale_rating">
  <img alt="SonarCloud Reliabilitye" src="https://sonarcloud.io/api/project_badges/measure?project=mtc1003_TF_Keras_TFG&metric=reliability_rating">
  <img alt="SonarCloud Vulnerabilities" src="https://sonarcloud.io/api/project_badges/measure?project=mtc1003_TF_Keras_TFG&metric=vulnerabilities">
  <img alt="SonarCloud Bugs" src="https://sonarcloud.io/api/project_badges/measure?project=mtc1003_TF_Keras_TFG&metric=bugs">
  <br>
  <a href="https://github.com/mtc1003/TF_Keras_TFG/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/mtc1003/TF_Keras_TFG"></a>
  <a href="https://github.com/mtc1003/TF_Keras_TFG/network/members"><img alt="GitHub forks" src="https://img.shields.io/github/forks/mtc1003/TF_Keras_TFG"></a>
  <a href="https://github.com/mtc1003/TF_Keras_TFG/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/mtc1003/TF_Keras_TFG"></a>
  <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/mtc1003/TF_Keras_TFG">
  <a href="https://github.com/mtc1003/TF_Keras_TFG/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/mtc1003/TF_Keras_TFG"></a> 
  <br/>
  <img alt="Non Comment Lines Of Code" src="https://sonarcloud.io/api/project_badges/measure?project=mtc1003_TF_Keras_TFG&metric=ncloc">
  <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/mtc1003/TF_Keras_TFG?color=purple&logo=github">
</div>

<div>
  <img alt="Logo UBU" src="https://investigacion.ubu.es/img/ubu_index_logo-8e75169cd8d76f7088bf88f023128f10.svg">
  <br>
  <h4 align="center"><strong>TFG Universidad de Burgos</strong></h4>
  <br>
  <h4>
    Autora: Miriam Torres Calvo
    <br>
    Tutor: Bruno Baruque Zanón
  </h4>
  <h4 align="center"><strong>Resumen</strong></h4>
  <br>
  <p>
    La tecnologia avanza continuamente y a velocidades que hace
    unos años eran inexplicables, contamos con su ayuda en diferentes
    entornos de la sociedad y cada vez en mayor medida. Pero también,
    contamos con situaciones en las cuáles está no se encuentra tan fácil
    de acceder, ya que no se cuenta con la facilidad para transportarla y
    así poder usarla en muchos mas lugares y mas comodamente.
    Así surge este proyecto, con la idea de poder usar un modelo de
    Machine Learning en dispositivos de Edge Computing, como puede
    ser la Raspberry Pi 3 de Raspberry.
    Para su desarrollo, se contara con el lenguaje Python y el modelo
    escogido para entrenar ha sido YOLO en su cuarta versión.
  </p>
  <h4 align="center"><strong>Primeros Pasos</strong></h4>
  
  ### Conda
  ```bash
  conda env create -f OD_MTC.yml
  conda activate ODMTC
  ```
  ### Pip
  ```bash
  pip install -r requirements.txt
  ```
  <h4 align="center"><strong>Descarga de modelos(YOLO)</strong></h4>
  https://drive.google.com/drive/folders/1JIn47yPThdm7FX0xs_fZnpg30M5jLfaK?usp=sharing

  <h4 align="center"><strong>Uso TensorFlow (Ejemplos)</strong></h4>
   
  ```bash
  # Ejecutar desde codigo\yolov4ToTensorFlow

  # Convertir modelo YOLO a Tensorflow
  python save_model.py --weights ./data/custom.weights --output ./checkpoints/custom-416 --input_size 416 --model yolov4

  # Detección de objetos en imagen
  python detect.py --weights .\checkpoints\custom-416\ --size 416 --image .\data\images\american_plates.jpg --output .\detections\images\license_plate_detction.png --model yolo

  # Detección de objetos en vídeo
  python detectVideo.py --weights .\checkpoints\custom-416\ --size 416 --model yolov4 --video .\data\video\cars.mp4 --output ./detections/videos/results_cars_ds.avi

  # Contabilización objetos en vídeo
  python objectTracker.py --weights .\checkpoints\custom-416 --video .\data\video\license_plate.mp4 --output .\detections\tracker.avi --model yolov4

  # Deteccion webcam
  python detectVideo.py --weights .\checkpoints\yolov4-416\ --size 416 --model yolov4 --video 0  --output ./detections/videos/webcam.avi
  ```

  <h4 align="center"><strong>Uso TensorFlow Lite (Ejemplos)</strong></h4>
   
  ```bash
  # Ejecutar desde codigo\yolov4ToTensorFlow

  # Convertir modelo YOLO a Tensorflow Lite
   python .\save_model_tflite.py --weights .\data\heads_v2.weights --output ./checkpoints/heads_tfl-416 --input_size 416 --model yolov4

   python .\convert_tflite.py --weights .\checkpoints\heads_tfl-416\ --output .\checkpoints\heads-416-int8.tflite --quantize_mode int8

  # Detección de objetos en imagen
  python detect.py --weights .\checkpoints\custom-416-int8.tflite --size 416 --image .\data\images\head_1.png --output .\detections\images\plate_tflite.png --model yolov4 --framework tflite --quality True

  # Detección de objetos en vídeo
  python detectVideo.py --weights .\checkpoints\yolov4-416-int8.tflite --size 416 --model yolov4 --video .\data\video\cars_r.mp4 --output ./detections/videos/results_cars.avi --dont_show --framework tflite --quality True

  # Contabilización objetos en vídeo
  python objectTracker.py --weights .\checkpoints\custom-416-int8.tflite --video .\data\video\license_plate.mp4 --output .\detections\tracker_tflite.avi --model yolov4 --framework tflite --quality True
  ```
<h4 align="center"><strong>Usar App OD-MTC</strong></h4>

```bash
# Ejecutar desde codigo\yolov4ToTensorFlow
python app.py
```
</div>
