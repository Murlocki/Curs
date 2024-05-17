import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

from view.main_window import MainWindow
from PyQt5.QtWidgets import QApplication
import sys
from ultralytics import YOLO
if __name__ == '__main__':


    app = QApplication(sys.argv)
    w = MainWindow().create()
    w=w.show()
    sys.exit(app.exec_())

    # image_path = r'C:\Users\kiril\Desktop\Curs2\TrolleysClassification-3\test\magnit_trolley'
    # #image_path = r'C:\Users\kiril\PycharmProjects\Curs\weights\IMG_20240406_165308_jpg.rf.2920c577af98032a17ceeff99778d9db.jpg'
    # model = YOLO(r'C:\Users\kiril\PycharmProjects\Curs\weights\yolo_w.pt')
    # results = model(image_path,save=True)
    #
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    #
    #
    # for result in results:
    #     img = cv2.imread(result.path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     print(result.path)
    #     boxes = result.boxes.cpu().numpy().xyxyn
    #     print(boxes)
    #     dh,dw,_ = img.shape
    #     print(dh,dw)
    #     boxes_draw = []
    #     for box in boxes:
    #         x1, y1, x2, y2 = int(box[0] * dw), int(box[1] * dh), int(box[2] * dw), int(box[3] * dh)
    #         print(x1,y1,x2,y2)
    #         img_crop = img[y1:y2,x1:x2]
    #         PIL_image = Image.fromarray(np.uint8(img_crop)).convert('RGB')
    #
    #         #plt.imshow(img_crop)
    #         #plt.show()
    #
    #         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #         print(device)
    #         model_classify = torchvision.models.inception_v3()
    #         model_classify.load_state_dict(torch.load(r'C:\Users\kiril\Desktop\Curs2\runs\inception v3\inception_batch_16_92\best\best.pt'))
    #
    #
    #         model_classify.eval()
    #         model_classify.to(device)
    #         with torch.no_grad():
    #             transform = transforms.Compose([
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         ])
    #             data = transform(PIL_image)
    #             input_batch = data.unsqueeze(0)
    #             input_batch = Variable(input_batch.to(device))
    #             output = model_classify(input_batch)
    #             print(output)
    #             probs = torch.softmax(output, dim=1)
    #             print(probs)
    #             top_k_probs, top_k_indices = probs.topk(3, dim=1)
    #             print(top_k_probs)
    #             print(top_k_indices)
    #
    #             probs_values = top_k_probs.cpu().detach().numpy()
    #             probs_indeces = top_k_indices.cpu().detach().numpy()
    #
    #             draw = ImageDraw.Draw(PIL_image)
    #             text = "leura:"+str(probs_values[0][np.where(probs_indeces[0]==0)]) +"\nmagnit:"+str(probs_values[0][np.where(probs_indeces[0]==1)])+"\nother:"+\
    #                    str(probs_values[0][np.where(probs_indeces[0]==2)])
    #             font = ImageFont.truetype('arial.ttf', 24)
    #             draw.text((10, 10), text, font=font, fill=(255, 255, 0))
    #             PIL_image.show()
    #
    #             boxes_draw.append([x1,y1,x2,y2,probs_indeces[0][0]])
    #
    #         image = cv2.imread(result.path)
    #         for box in boxes_draw:
    #             # Draw a rectangle
    #             color = (0, 255, 0) if box[4]==0 else (0,0,255)
    #             cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
    #
    #             # Display the output
    #         cv2.imshow('Image', image)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
