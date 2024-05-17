from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn import metrics
from main import resiz


def gradientBoostClass(train_data,classes,test_data,classes_test):
    best_acc = 0
    best_mat = 0
    best_n = 0
    for n in range(100,1100,100):
        model = GradientBoostingClassifier(n_estimators=n)
        model.fit(train_data,classes)
        y_pred = model.predict(test_data)
        accuracy = model.score(test_data, classes_test)
        print(y_pred, classes_test)
        print(f"Accuracy {accuracy}")
        print(model.classes_)
        if accuracy>best_acc:
            best_mat = ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(y_pred, classes_test),
                                      display_labels=model.classes_)
            best_n = n
            best_acc = accuracy
    return best_acc,best_n,best_mat
def processGrad():
    accuraccies = []
    best_params = []
    best_matrixes = []
    for size in [(32, 32)]:
        print(size)
        acc,best_n,best_matrix = gradientBoostClass(*resiz(size[0],size[1],0))
        accuraccies.append(acc)
        print(best_params)
        best_params.append(best_n)
        best_matrixes.append(best_matrix)

    plt.plot([32, 64, 128, 224], accuraccies)
    plt.title("Accuracies")
    plt.xlabel("value of image size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.plot([32, 64, 128, 224], best_params)
    plt.title("Best n iterations")
    plt.xlabel("image size")
    plt.ylabel("value of n iterations")
    plt.legend()
    plt.show()

    print(accuraccies)
    for matr in best_matrixes:
        matr.plot()
        plt.show()
processGrad()