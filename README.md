# Vehicle Classification VGG7 - 4 Classes
Bike | Bus | Car | Truck

## Full Pipeline Completed (10 Steps)
1. Data Collection - 400 images from 50k dataset
2. Labeling - 1_labelling.py -> labels.csv
3. EDA - 3_eda.py -> class_balance.png
4. Preprocessing - 4_preprocessing.py -> 320 train / 80 val, 128x128, /255
5. Augmentation - 5_augmentation.py -> 960 train (rotation20, zoom0.2, shift0.1)
6. Train - 6_train.py -> VGG7 (4.48M params), Batch32, 10 epochs, 30/30 steps, 72.19% train
7. Evaluation - 7_evaluation.py -> confusion matrix, 60% accuracy
8. Testing - 8_testing.py -> tested val images (car 49% vs bus 35%)
9. Performance Evaluation - 9_performanceevaluation.py -> performance_curves.png, confusion_matrix_final.png
10. GitHub - Ready to deploy

## Results
- Accuracy: 60% (80 val)
- Macro avg: 0.60
- Best: bike 88% precision
- Graphs: dataset/performance_curves.png, dataset/confusion_matrix_final.png

## Run
pip install -r requirements.txt
python src/3_eda.py
python src/6_train.py