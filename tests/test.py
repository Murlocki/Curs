from ultralytics import YOLO
import os
from matplotlib import pyplot as plt

#from roboflow import Roboflow
#rf = Roboflow(api_key="ljQZ11ZZYYC2aGnlcJr0")
#project = rf.workspace("kirill-kornilov-kn3yx").project("shopping-trolley-kn5tj")
#dataset = project.version(2).download("yolov8")


if __name__=='__main__':

    def size():
        r = []
        for filename in os.listdir('..\weights'):
            n = []
            f = os.path.join(r'..\weights',  filename)
            model = YOLO(f)
            result = model.val(data = r'C:\Users\kiril\PycharmProjects\Curs\tests\shopping-trolley-2\data.yaml')
            print(result.box.map,
            result.box.map50,
            result.box.map75,
            )
            n.append(result.box.map)
            try:
                n.append(int(filename[filename.index('_')+1:filename.index('.')]))
            except Exception:
                n.append(r[len(n)-1][1]+100)
            r.append(n)
        r.sort(key = lambda x:x[1])
        print(r)
        plt.plot([i[1] for i in r],[i[0] for i in r])
        plt.xlabel('Размер датасета')
        plt.ylabel('mAp50-95')
        plt.show()

        n = [5,8,10,16,32,64]
        plt.plot(n, [i[0] for i in r])
        plt.xlabel('Размер датасета')
        plt.ylabel('mAp50-95')
        plt.show()


    size()


