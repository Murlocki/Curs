import sklearn.preprocessing
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import GridSearchCV
from ultralytics import YOLO


# from roboflow import Roboflow
# rf = Roboflow(api_key="ljQZ11ZZYYC2aGnlcJr0")
# project = rf.workspace("kirill-kornilov-kn3yx").project("trolleysclassification")
# version = project.version(3)
# dataset = version.download("folder")



#Load a model
if __name__ == '__main__':
     model = YOLO(r'C:\Users\kiril\Desktop\Curs2\runs\classify\batch_16_0954\weights\best.pt')
     results = model(r'C:\Users\kiril\Desktop\Curs2\TrolleysClassification-3\test\magnit_trolley\IMG_20240406_164604-1-_png.rf.9c4a48f93c88df4ee95fb6ec09e39a8a.jpg',save=True)


from skimage.feature import hog
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import os
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics, svm


def create_array(path,x,y,class_name,classes=[],cluster_param=0):
    data=[]
    for entry in glob.glob(path):
        img = np.array(mpimg.imread(entry))
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = cv2.resize(img1, (x, y))
        img2 = sklearn.preprocessing.normalize(img2.reshape(x*y,3)).reshape(x,y,3)
        if cluster_param==1:
            img2 = clusterize(img2,x,y)
        if cluster_param==2:
            img2=np.vstack((img2.reshape([x,y,3]),clusterize(img2,x,y)))
        img3 = np.reshape(img2,[img2.shape[0]*img2.shape[1]*3],order="F")
        data.append(img3)
        classes.append(class_name)
    return data,classes

def resiz(x,y,cluster_param=0):
    classes=[]
    data_leura,classes= create_array(r"C:\Users\kiril\Desktop\Curs2\TrolleysClassification-3\train\leura_trolley\*",x,y,"leura",cluster_param=cluster_param,classes=classes)
    data_magnit,classes = create_array(r"C:\Users\kiril\Desktop\Curs2\TrolleysClassification-3\train\magnit_trolley\*",x,y,"magnit",cluster_param=cluster_param,classes=classes)
    other,classes = create_array(r"C:\Users\kiril\Desktop\Curs2\TrolleysClassification-3\train\other\*",x,y,"other",cluster_param=cluster_param,classes=classes)
    # print(data_leura)
    # print(data_magnit)
    # print(other)
    train_data = data_leura+data_magnit+other
    print(len(train_data),len(classes))

    classes_test=[]
    data_leura_test,classes_test = create_array(r"C:\Users\kiril\Desktop\Curs2\TrolleysClassification-3\val\leura_trolley\*",x,y,"leura",cluster_param=cluster_param,classes=classes_test)
    data_magnit_test,classes_test = create_array(r"C:\Users\kiril\Desktop\Curs2\TrolleysClassification-3\val\magnit_trolley\*",x,y,"magnit",cluster_param=cluster_param,classes=classes_test)
    other_test,classes_test = create_array(r"C:\Users\kiril\Desktop\Curs2\TrolleysClassification-3\val\other\*",x,y,"other",cluster_param=cluster_param,classes=classes_test)
    print(data_leura_test)
    print(data_magnit_test)
    print(other_test)

    test_data=data_leura_test+data_magnit_test+other_test

    print(classes)
    print(classes_test)
    return train_data,classes,test_data,classes_test
def clusterize(img2,x,y):
    img3 = img2.reshape(x*y,3)
    n = 60
    k_means = KMeans(n_clusters=n,n_init='auto')
    model = k_means.fit(img3)
    centroids = k_means.cluster_centers_
    labels = k_means.labels_
    res_labels = centroids[labels]
    result_image = res_labels.reshape(x, y, 3)
    result_image = result_image.astype(np.uint8)


    return np.round(result_image)




