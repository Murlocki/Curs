
import matplotlib.pyplot as plt
from sklearn import metrics, svm
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV


def svm_process(train_data,classes,test_data,classes_test):
    param_grid = {
        'C':[0.1],
        'kernel': [ 'poly']
    }
    svc = svm.SVC(probability=True)
    model = GridSearchCV(svc, param_grid,refit=True)
    model.fit(train_data, classes)
    best_params = model.best_params_
    print(best_params)
    print(323)
    y_pred = model.predict(test_data)
    accuracy = model.score(test_data,classes_test)
    print(y_pred,classes_test)
    print(f"Accuracy {accuracy}")
    disp=ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(y_pred, classes_test),display_labels=model.classes_)
    return accuracy,disp,best_params
def processSvm():

    accuraccies = []
    best_params = []
    best_matrixes = []
    for size in [(32, 32),(64,64),(128,128),(224,224)]:
        print(size)
        acc,best_matrix,best_param = svm_process(*resiz(size[0],size[1],0))
        accuraccies.append(acc)
        print(best_params)
        best_params.append(best_param)
        best_matrixes.append(best_matrix)

    plt.plot([32, 64, 128, 224], [0.66,0.67,0.65,0.6])
    plt.title("Accuracies")
    plt.xlabel("value of image size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.plot([32, 64, 128, 224], [1,0.1,0.1,1])
    plt.title("Best c iterations")
    plt.xlabel("image size")
    plt.ylabel("value of c iterations")
    plt.legend()
    plt.show()

    data = {'32':'poly','64':'rbf','128':'poly','224':'poly'}
    plt.bar(data.keys(),data.values())
    plt.title("Best kernel iterations")
    plt.xlabel("image size")
    plt.ylabel("value of kernel iterations")
    plt.legend()
    plt.show()

    print(accuraccies)
    for matr in best_matrixes:
        matr.plot()
        plt.show()
processSvm()