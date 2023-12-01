from ultralytics import YOLO



#from roboflow import Roboflow
#rf = Roboflow(api_key="ljQZ11ZZYYC2aGnlcJr0")
#project = rf.workspace("kirill-kornilov-kn3yx").project("shopping-trolley-kn5tj")
#dataset = project.version(2).download("yolov8")


if __name__=='__main__':
    model = YOLO('../weights/yolo_w.pt')
    result = model.val(data = r'C:\Users\kiril\PycharmProjects\Curs\tests\shopping-trolley-2\data.yaml')
    print(result.box.map,
    result.box.map50,
    result.box.map75,
    result.box.maps, )