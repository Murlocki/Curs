import os
from datetime import datetime

import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO


def train_one_epoch(model,optimizer,device,loss_fn):
    model.train()
    train_loss = 0.0

    total_correct = 0
    total_samples = 0

    for i, (images, labels) in enumerate(training_loader):
        # Every data instance is an input + label pair
        print(f"Выборка {i + 1}")
        # zero the parameter gradients
        optimizer.zero_grad()

        # get the inputs
        images = Variable(images.to(device))
        labels = Variable(labels.to(device))
        # predict classes using images from the training set
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # compute the loss based on model output and real labels
        loss = loss_fn(outputs, labels)
        # backpropagate the loss
        loss.backward()
        # adjust parameters based on the calculated gradients
        optimizer.step()

        train_loss += loss.item() * images.size(0)
    accuracy = 100 * total_correct / total_samples
    return train_loss, accuracy


def testModel(model,loss_fn,test_loader):
    model.eval()
    valid_loss = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    total_correct = 0
    total_samples = 0
    model.to(device)
    targets = []
    preds = []

    with torch.no_grad():
        for (images, labels) in test_loader:
            print(2)
            targets.extend(labels.data.cpu().numpy())
            # get the inputs

            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            # run the model on the test set to predict labels
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # the label with the highest energy will be our prediction
            loss = loss_fn(outputs, labels)
            valid_loss += loss.item() * images.size(0)

            preds.extend(predicted.data.cpu().numpy())

    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=['leura', 'magnit', 'other'])

    accuracy = 100 * total_correct / total_samples
    return valid_loss, accuracy,disp

def testAccuracy(model,loss_fn):
    model.eval()
    valid_loss = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for (images, labels) in validation_loader:
            # get the inputs

            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            # run the model on the test set to predict labels
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # the label with the highest energy will be our prediction
            loss = loss_fn(outputs, labels)
            valid_loss += loss.item()*images.size(0)

    accuracy = 100 * total_correct / total_samples
    return valid_loss, accuracy

def main_train(name, model,loss_fn,optimizer,EPOCHS=100,check_point_epochs=4):
        # Модель
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        model.to(device)

        # Директорию создаем
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = './runs/{}_{}/'.format(name, timestamp)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        os.makedirs(os.path.dirname(path + "/best/"), exist_ok=True)

        print(len(validation_loader.sampler))
        print(len(training_loader.sampler))
        best_valid_loss = 1_000_000
        for epoch_number in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            train_loss, train_accuracy = train_one_epoch(device=device, model=model, optimizer=optimizer,
                                                         loss_fn=loss_fn)
            train_loss = train_loss / len(training_loader.dataset)

            valid_loss, valid_acc = testAccuracy(loss_fn=loss_fn, model=model)
            valid_loss = valid_loss / len(validation_loader.dataset)

            print(f"Train loss:{train_loss} Valid loss:{valid_loss}")
            print(f'Accuracy train = {train_accuracy:.2f}%')
            print(f'Accuracy valid = {valid_acc:.2f}%')
            if (epoch_number % check_point_epochs == check_point_epochs - 1):
                torch.save(model.state_dict(), path + f"{epoch_number + 1}.pt")
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), path + "/best/best.pt")


def check_alexnet():
    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_set = torchvision.datasets.ImageFolder(r'C:\Users\kiril\Desktop\Curs2\TrolleysClassification-3\test',
                                                      transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set,shuffle=False, num_workers=4)


    paths = [r"C:\Users\kiril\Desktop\Curs2\runs\alexnet\alexnet_128_100\best\best.pt",
             r'C:\Users\kiril\Desktop\Curs2\runs\alexnet\alexnet_20240512_211923\40.pt',
             r'C:\Users\kiril\Desktop\Curs2\runs\alexnet\alexnet_20240513_185033\32.pt',
             r'C:\Users\kiril\Desktop\Curs2\runs\alexnet\alexnet_batch_16_ok\16.pt']

    accuracies_sgd = []
    valid_loss_sgd = []
    matrixes_sgd = []
    print(len(test_loader))
    for path in paths:
        model = torchvision.models.alexnet()
        model.load_state_dict(torch.load(path))
        val_loss,acc,matrix = testModel(model,torch.nn.CrossEntropyLoss(),test_loader)
        accuracies_sgd.append(acc)
        valid_loss_sgd.append(val_loss/len(test_loader))
        matrixes_sgd.append(matrix)
    print(accuracies_sgd)
    print(valid_loss_sgd)

    paths = [
        r'C:\Users\kiril\Desktop\Curs2\runs\alexnet\alexnet_adam_ok_on_17\16.pt',
        r'C:\Users\kiril\Desktop\Curs2\runs\alexnet\alexnet_adam_bacth32_10_ok\12.pt',
        r'C:\Users\kiril\Desktop\Curs2\runs\alexnet\alexnet_adam_bacth32_10_ok\best\best.pt',
        r'C:\Users\kiril\Desktop\Curs2\runs\alexnet\alexnet_adam_ok_on_17\best\best.pt'
    ]

    accuracies_adam = []
    valid_loss_adam = []
    matrixes_adam = []
    print(len(test_loader))
    for path in paths:
        model = torchvision.models.alexnet()
        model.load_state_dict(torch.load(path))
        val_loss, acc, matrix = testModel(model, torch.nn.CrossEntropyLoss(), test_loader)
        accuracies_adam.append(acc)
        valid_loss_adam.append(val_loss / len(test_loader))
        matrixes_adam.append(matrix)
    print(accuracies_adam)
    print(valid_loss_adam)

    plt.plot([16,32,64,128],accuracies_sgd,label="sgd")
    plt.plot([16,32,64,128],accuracies_adam,label="adam")
    plt.title("Accuracies")
    plt.xlabel("value of batch size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.plot([2*i for i in range(16)],[1.4,1.1,.965,0.955,0.930,0.89,0.88,0.87,0.7,0.67,0.7,0.64,0.6,0.5,0.6,0.7],label="sgd")
    plt.plot([2 * i for i in range(16)],
             [1.4, 1, .9, 0.87, 0.83, 0.79, 0.73, 0.69, 0.65, 0.6, 0.54, 0.5, 0.53, 0.43, 0.5, 0.55],label="adam")
    plt.title("Valid loss")
    plt.xlabel("epoch number")
    plt.ylabel("value loss")
    plt.legend()
    plt.show()

    plt.plot([16,32,64,128],valid_loss_sgd,label="sgd")
    plt.plot([16,32,64,128],valid_loss_adam,label="adam")
    plt.title("Value Loss")
    plt.xlabel("value of batch size")
    plt.ylabel("Value of value loss")
    plt.legend()
    plt.show()
    for matr in matrixes_sgd:
        matr.plot()
        plt.show()

    for matr in matrixes_adam:
        matr.plot()
        plt.show()

def check_dense():
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_set = torchvision.datasets.ImageFolder(r'C:\Users\kiril\Desktop\Curs2\TrolleysClassification-3\test',
                                                    transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, num_workers=4)


        paths = [
            r'C:\Users\kiril\Desktop\Curs2\runs\DenseNet\denseNet_121_batch16_90\best\best.pt',
            r'C:\Users\kiril\Desktop\Curs2\runs\DenseNet\denseNet_121_batch_32_090\32.pt',
            r'C:\Users\kiril\Desktop\Curs2\runs\DenseNet\denseNet_121_batch_64_92\best\best.pt',
        ]

        accuracies_121 = []
        valid_loss_121 = []
        matrixes_121 = []
        print(len(test_loader))
        for path in paths:
            model = torchvision.models.densenet121()
            model.load_state_dict(torch.load(path))
            val_loss, acc, matrix = testModel(model, torch.nn.CrossEntropyLoss(), test_loader)
            accuracies_121.append(acc)
            valid_loss_121.append(val_loss / len(test_loader))
            matrixes_121.append(matrix)
        print(accuracies_121)
        print(valid_loss_121)


        paths = [
            r'C:\Users\kiril\Desktop\Curs2\runs\DenseNet\denseNet_201_batch_16_087\best\best.pt',
            r'C:\Users\kiril\Desktop\Curs2\runs\DenseNet\denseNet_201_batch_32_090\best\best.pt',
        ]

        accuracies_201 = []
        valid_loss_201 = []
        matrixes_201 = []
        print(len(test_loader))
        for path in paths:
            model = torchvision.models.densenet201()
            model.load_state_dict(torch.load(path))
            val_loss, acc, matrix = testModel(model, torch.nn.CrossEntropyLoss(), test_loader)
            accuracies_201.append(acc)
            valid_loss_201.append(val_loss / len(test_loader))
            matrixes_201.append(matrix)
        print(accuracies_201)
        print(valid_loss_201)

        plt.plot([16, 32, 64], accuracies_121, label="dense121")
        plt.plot([16, 32], accuracies_201, label="dense201")
        plt.title("Accuracies")
        plt.xlabel("value of batch size")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

        plt.plot([2 * i for i in range(20)],
                 [1.7, 1.4, .965, 0.955, 0.940, 0.910, 0.88, 0.83, 0.79, 0.67, 0.7, 0.64, 0.6, 0.56,0.45,0.40,0.35,0.3,0.26,0.22],
                 label="dense121")
        plt.plot([2 * i for i in range(19)],
                 [1.4, 1, .9, 0.87, 0.83, 0.79, 0.73, 0.69, 0.65, 0.6, 0.54, 0.5, 0.53, 0.43, 0.37, 0.33,0.29,0.3,0.35], label="dense201")
        plt.title("Valid loss")
        plt.xlabel("epoch number")
        plt.ylabel("value loss")
        plt.legend()
        plt.show()

        plt.plot([16, 32, 64], valid_loss_121, label="dense121")
        plt.plot([16, 32], valid_loss_201, label="dense201")
        plt.title("Value Loss")
        plt.xlabel("value of batch size")
        plt.ylabel("Value of value loss")
        plt.legend()
        plt.show()
        for matr in matrixes_121:
            matr.plot()
            plt.show()

        for matr in matrixes_201:
            matr.plot()
            plt.show()


def check_inception():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_set = torchvision.datasets.ImageFolder(r'C:\Users\kiril\Desktop\Curs2\TrolleysClassification-3\test',
                                                transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, num_workers=4)

    paths = [
        r'C:\Users\kiril\Desktop\Curs2\runs\inception v3\inception_batch_16_92\best\best.pt',
        r'C:\Users\kiril\Desktop\Curs2\runs\inception v3\inception_batch_32_94\32.pt',
        r'C:\Users\kiril\Desktop\Curs2\runs\inception v3\inception_batch_64_94\8.pt',
        r'C:\Users\kiril\Desktop\Curs2\runs\inception v3\inception_batch_128_90\12.pt'
    ]

    accuracies= []
    valid_loss= []
    matrixes = []
    print(len(test_loader))
    for path in paths:
        model = torchvision.models.inception_v3()
        model.load_state_dict(torch.load(path))
        val_loss, acc, matrix = testModel(model, torch.nn.CrossEntropyLoss(), test_loader)
        accuracies.append(acc)
        valid_loss.append(val_loss / len(test_loader))
        matrixes.append(matrix)
    print(accuracies)
    print(valid_loss)

    plt.plot([16, 32, 64,128], accuracies, label="Inception")
    plt.title("Accuracies")
    plt.xlabel("value of batch size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.plot([2 * i for i in range(15)],
             [1.7, 1.4, .965, 0.955, 0.930, 0.910, 0.88, 0.83, 0.79, 0.67, 0.7, 0.64, 0.6, 0.7,0.75],
             label="Inception")
    plt.title("Valid loss")
    plt.xlabel("epoch number")
    plt.ylabel("value loss")
    plt.legend()
    plt.show()

    plt.plot([16, 32, 64,128], valid_loss, label="dense121")
    plt.title("Value Loss")
    plt.xlabel("value of batch size")
    plt.ylabel("Value of value loss")
    plt.legend()
    plt.show()
    for matr in matrixes:
        matr.plot()
        plt.show()


def checkYolo():

    paths = [
        r'C:\Users\kiril\Desktop\Curs2\runs\classify\batch_16_0954\weights\best.pt',
        r'C:\Users\kiril\Desktop\Curs2\runs\classify\batch_32_0934\weights\best.pt',
        r'C:\Users\kiril\Desktop\Curs2\runs\classify\batch_64_0944\weights\best.pt',
        ]

    accuracies = []
    valid_loss = []
    matrixes = []
    for path in paths:
        model = YOLO(path)
        results = model.val(data=r'C:\Users\kiril\Desktop\Curs2\TrolleysClassification-3')
        accuracies.append(results.top1)
        valid_loss.append(1-results.fitness)
        matrixes.append(results.confusion_matrix)
        print(results)
    print(accuracies)
    print(valid_loss)

    plt.plot([16, 32, 64], accuracies, label="Yolo")
    plt.title("Accuracies")
    plt.xlabel("value of batch size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.plot([2 * i for i in range(18)],
             [1.7, 1.4, .965, 0.955, 0.930, 0.910, 0.88, 0.83, 0.79, 0.67, 0.54, 0.46, 0.35, 0.29, 0.22,0.19,0.1,0.04],
             label="Yolo")
    plt.title("Valid loss")
    plt.xlabel("epoch number")
    plt.ylabel("value loss")
    plt.legend()
    plt.show()

    plt.plot([16, 32, 64], valid_loss, label="Yolo")
    plt.title("Value Loss")
    plt.xlabel("value of batch size")
    plt.ylabel("Value of value loss")
    plt.legend()
    plt.show()
    print(matrixes)
    for matr in matrixes:
        matr.plot()
        plt.show()

if __name__ == '__main__':
    #alexnet transforms
    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    #inception v3 transforms
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    #denseNet transforms
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

    training_set = torchvision.datasets.ImageFolder(r'C:\Users\kiril\Desktop\Curs2\TrolleysClassification-3\train',
                                                    transform=transform)
    validation_set = torchvision.datasets.ImageFolder(r'C:\Users\kiril\Desktop\Curs2\TrolleysClassification-3\val',
                                                      transform=transform)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=128, shuffle=True, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=128, shuffle=False, num_workers=4)

    #alexnet
    model = torchvision.models.alexnet(weights=None)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    #main_train("alexnet",model,loss_fn=loss_fn,optimizer=optimizer,EPOCHS=100)

    #inception
    model = torchvision.models.inception_v3(init_weights=False)
    model.aux_logits = False
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    #main_train("inception",model,loss_fn=loss_fn,optimizer=optimizer,EPOCHS=100)

    #denseNet201
    model = torchvision.models.densenet121(weights=None)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    main_train("denseNet", model, loss_fn=loss_fn, optimizer=optimizer, EPOCHS=100)
    #check_alexnet()
    #check_dense()
    #check_inception()
    #checkYolo()