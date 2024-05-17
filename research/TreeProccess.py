
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

from main import resiz


def proccessTree():
    accuraccies = []
    best_n_s = []
    best_matrixes = []
    for size in [(32,32),(64,64),(128,128),(224, 224)]:
        print(size)
        acc,b_n,matrix = random_tree(*resiz(size[0], size[1], cluster_param=0))
        accuraccies.append(acc)
        best_n_s.append(b_n)
        best_matrixes.append(matrix)

    plt.plot([32, 64, 128, 224], accuraccies)
    plt.title("Accuracies")
    plt.xlabel("value of image size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.plot([32, 64, 128, 224], best_n_s)
    plt.title("Best n iterations")
    plt.xlabel("image size")
    plt.ylabel("value of n iterations")
    plt.legend()
    plt.show()

    print(accuraccies)
    for matr in best_matrixes:
        matr.plot()
        plt.show()
def random_tree(train_data,classes,test_data,classes_test):
    best_accuracy = 0
    best_n = 0
    best_matrix = None
    for i in range(100,1100,100):
        model = RandomForestClassifier(n_estimators=i, criterion='gini')
        model.fit(train_data, classes)
        y_pred = model.predict(test_data)
        accuracy = model.score(test_data, classes_test)
        if accuracy>best_accuracy:
            best_n = i
            best_accuracy = accuracy
            best_matrix =  ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(y_pred, classes_test),
                                      display_labels=model.classes_)
        print(f"Accuracy {accuracy}")
        # disp = ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(y_pred, classes_test),
        #                               display_labels=model.classes_)
        # disp.plot()
        # plt.show()
    return best_accuracy,best_n,best_matrix