# Surveillance system designed to enhance predictive capabilities in identifying potential gun-related incidents

Leveraging state-of-the-art machine learning algorithms, this project proposes a system to analyze CCTV (Closed Circuit Television) footage to predict the future appearance of guns before they are visibly drawn. It addresses this in three parts, starting with state of the art weapon's detection, followed by collection of dataset that includes CCTV footage with and without guns and lastly integrating these two to predict the future appearance of a gun. The state of the art weapon's detection works well but gives sparse prediction for CCTV data and the intent prediction seems to be feasible with validation loss decreasing.  Further improvement and future integration of such components in the modern-day surveillance systems, to forecast potential threats and provide real-time alerts to law enforcement agencies, not limited to just the CCTV cameras (including for example: Body-cams used by law enforcement authorities) could thus be feasible through collaboration with law enforcement and community stakeholders.

This repository organizes the code from the project in 2 main directories:

```
├── Intent Prediction
│   ├── baseline_model.ipynb
│   └── gun_prediction_MobileNet+EfficientNet.ipynb
├── Weapon's Detection
│   ├── darkvision_finetune
│   ├── finetune-sam.ipynb
│   ├── finetuning_v8.ipynb
│   ├── yolo-mamba
│   ├── yolo_v8_best.pt
│   └── yolov8-experiments
└── README.md
```

Intent prediction contains all the code to prediction the future appereance of guns with a prior of 5 seconds. Weapon's detection contains multiple implementations of object detection pipelines including DarkVision, YOLO v8 and YOLO-Mamba. 


