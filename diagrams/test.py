from ultralytics import YOLO
import os
from matplotlib import pyplot as plt

#from roboflow import Roboflow
#rf = Roboflow(api_key="ljQZ11ZZYYC2aGnlcJr0")
#project = rf.workspace("kirill-kornilov-kn3yx").project("shopping-trolley-kn5tj")
#dataset = project.version(2).download("yolov8")


if __name__=='__main__':
    plt.plot([32,64,128,224],[0.625,0.66,0.645,0.645],label="k-neighboors Euclidian")
    plt.plot([32, 64, 128, 224], [0.6, 0.59, 0.61, 0.61], label="k-neighboors Manhattan")
    plt.plot([32, 64, 128, 224], [0.695, 0.691, 0.696, 0.71], label="random forest")
    plt.plot([32, 64, 128, 224], [0.66, 0.67, 0.65, 0.6], label="SVM")
    plt.plot([32, 64, 128, 224], [0.6, 0.65, 0.63, 0.635], label="Gradient Boosting")
    plt.xlabel("image size")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    plt.plot([16,32, 64, 128], [0.81, 0.76, 0.73, 0.83], label="AlexNet SGD")
    plt.plot([16,32, 64, 128], [0.79, 0.81, 0.76, 0.75], label="AlexNet Adam")
    plt.plot([16,32, 64, 128], [0.79, 0.87, 0.86, 0.82], label="Inception")
    plt.plot([16,32, 64, 128], [0.87, 0.915, 0.9, 0.85], label="Dense201")
    plt.plot([16,32, 64, 128], [0.9, 0.95, 0.91, 0.88], label="Dense121")
    plt.plot([16, 32, 64, 128], [0.955, 0.935, 0.945, 0.93], label="YOLO")
    plt.xlabel("batch size")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


    # def size():
        #r = []
        #for filename in os.listdir('..\weights'):
         #   n = []
          #  f = os.path.join(r'..\weights',  filename)
           # model = YOLO(f)
            #result = model.val(data = r'C:\Users\kiril\PycharmProjects\Curs\diagrams\shopping-trolley-2\data.yaml')
            #print(result.box.map,
            #result.box.map50,
            #result.box.map75,
            #)
            #n.append(result.box.map)
            #try:
             #   n.append(int(filename[filename.index('_')+1:filename.index('.')]))
            #except Exception:
             #   n.append(r[len(n)-1][1]+100)
            #r.append(n)
        #r.sort(key = lambda x:x[1])
        #print(r)


        # plt.plot([6,10,15,30,40,110,114,130,135,138,140,147,150,160,170,180,200],
        #          [0.2816883736586792,
        #           0.292,
        #           0.3316883736586792,
        #           0.4216883736586792,
        #           0.4816883736586792,
        #           0.7457207525022116,
        #           0.7546734586361138,
        #           0.7658112943214042,
        #           0.7457207525022116,
        #           0.7957207525022116,
        #           0.8390217461118581,
        #           0.8393115266297688,
        #           0.8447699599307903,
        #           .8412767648723529,
        #           0.8514465910037646,
        #           0.8410150756243681,
        #           0.8460918753295774])
        # plt.grid()
        # plt.xlabel('Количество эпох')
        # plt.ylabel('mAp50-95')
        # plt.show()
        #
        # n = [600,1000,2000,2500,3000,3500,4000,4500,5000,6000,6714]
        # plt.plot(n, [
        #     0.4216883736586792,
        #     0.4216883736586792,
        #     0.5457207525022116,
        #     0.5957207525022116,
        #     0.6457207525022116,
        #     0.7057207525022116,
        #     0.7457207525022116,
        #     0.7546734586361138,
        #     0.7957207525022116,
        #     0.8390217461118581,
        #     0.8514465910037646
        # ])
        # plt.grid()
        # plt.xlabel('Размер датасета')
        # plt.ylabel('mAp50-95')
        # plt.show()
        #
        # n = [160, 320, 640, 1280]
        # plt.plot(n, [
        #     0.5457207525022116,
        #     0.7057207525022116,
        #     0.7457207525022116,
        #     0.8514465910037646,
        # ])
        # plt.grid()
        # plt.xlabel('Размер изображения')
        # plt.ylabel('mAp50-95')
        # plt.show()
        #
        # n = [4, 6, 7, 9, 10, 12, 16, 20, 32, 48, 64]
        # plt.plot(n, [
        #     0.416883736586792,
        #     0.4416883736586792,
        #     0.5857207525022116,
        #     0.6057207525022116,
        #     0.6557207525022116,
        #     0.7257207525022116,
        #     0.7557207525022116,
        #     0.7746734586361138,
        #     0.7957207525022116,
        #     0.8390217461118581,
        #     0.8514465910037646
        # ])
        # plt.grid()
        # plt.xlabel('Размер выборки')
        # plt.ylabel('mAp50-95')
        # plt.show()




