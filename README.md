# Covid_XRay_Classifier

Multi-Class CNN image classifier on Chest Xray images provided by University of Montreal

Overview:
Uses data from https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset

Features:
Image Augmentation to account for 'small' size of dataset.
Sequential Model with 2 CNNs and Batch descent
Classfication accuracy is stable >80% for current version.


To Do:
Address Loss function creep up
Data Viz
Compare to basic scikit method
try deeper model with fewer params (Large Dense layer?)
Find other Xray sets to compare performace
EXTRACT FEATURES-->Heatmap? Filters?
