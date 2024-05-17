
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from main import resiz


def process_neighbours(train_data,classes,test_data,classes_test):
    max_neighbours = 30

    x_axis_k_points = []
    acc_euclidian=[]
    conf_matrix_euc=[]
    best_acc_euc = 0
    best_acc_man = 0
    best_euc_k = 0
    best_man_k = 0

    best_conf_matrix_euc = None
    best_conf_matrix_man = None

    for k in range(max_neighbours):
        knn_euc = KNeighborsClassifier(n_neighbors=k+1)
        knn_euc.fit(train_data,classes)

        #Предсказываем
        pred_labels_euc = knn_euc.predict(test_data)
        accurracy = knn_euc.score(test_data,classes_test)
        acc_euclidian.append(accurracy)

        if accurracy>best_acc_euc:
            best_acc_euc=accurracy
            best_euc_k=k
            best_conf_matrix_euc = metrics.confusion_matrix(classes_test,pred_labels_euc,labels=knn_euc.classes_)
        #conf_matrix_euc.append(metrics.confusion_matrix(classes_test,pred_labels_euc,labels=knn_euc.classes_))


        x_axis_k_points.append(k+1)
        print(2)
    acc_man=[]
    conf_matrix_man=[]
    for k in range(max_neighbours):
        knn_man = KNeighborsClassifier(n_neighbors=k+1,p=1)
        knn_man.fit(train_data,classes)

        #Предсказываем
        pred_labels_man = knn_man.predict(test_data)
        accurracy = knn_man.score(test_data,classes_test)
        acc_man.append(accurracy)

        #conf_matrix_man.append(metrics.confusion_matrix(classes_test,pred_labels_man,labels=knn_man.classes_))

        if best_acc_man<accurracy:
            best_acc_man=accurracy
            best_man_k = k
            best_conf_matrix_man = metrics.confusion_matrix(classes_test,pred_labels_man,labels=knn_man.classes_)
        print(1)

    # for i in range(len(acc_man)):
    #     print("For k = ",i+1,"acc=",acc_man[i],'conf matr=',conf_matrix_man[i],end='\n')
    #     disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_man[i],display_labels = knn_man.classes_)
    #     disp.plot()
    #     plt.show()
    # for i in range(len(acc_euclidian)):
    #     print("For k = ",i+1,"acc=",acc_euclidian[i],'conf matr=',conf_matrix_euc[i],end='\n')
    #     disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_euc[i], display_labels=knn_euc.classes_)
    #     disp.plot()
    #     plt.show()

    # plt.plot(x_axis_k_points,acc_euclidian,label="Euclidian")
    # plt.plot(x_axis_k_points,acc_man,label="Manhattan")
    # plt.title("Accuracies")
    # plt.xlabel("value of k")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.show()

    return best_acc_euc,best_acc_man,best_euc_k,best_man_k,best_conf_matrix_euc,best_conf_matrix_man,knn_euc.classes_,knn_man.classes_
def proccess_kneigbh():
    accuracy_euc = []
    accuracy_man = []
    k_euc = []
    k_man = []
    euc_matrix = []
    man_matrix = []
    for size in [(32,32),(64,64),(128,128),(224,224)]:
        print(size)
        best_euc,best_man,best_euc_k,best_man_k,best_mat_euc,best_mat_man,k_euc_classes,k_man_classes = process_neighbours(*resiz(size[0],size[1],cluster_param=0))
        accuracy_euc.append(best_euc)
        accuracy_man.append(best_man)
        k_euc.append(best_euc_k)
        k_man.append(best_man_k)
        euc_matrix.append(best_mat_euc)
        man_matrix.append(best_mat_man)
    plt.plot([32,64,128,224],accuracy_euc)
    plt.plot([32,64,128,224],accuracy_man)
    plt.show()
    #16 5 8 5
    #4 2 2 2
    plt.plot([32,64,128,224],k_euc)
    plt.plot([32,64,128,224],k_man)
    plt.show()
    print(accuracy_euc)
    print(accuracy_man)
    for i in range(len(euc_matrix)):
        disp = ConfusionMatrixDisplay(confusion_matrix=euc_matrix[i], display_labels=k_euc_classes)
        disp.plot()
        plt.show()
    for i in range(len(man_matrix)):
        disp = ConfusionMatrixDisplay(confusion_matrix=man_matrix[i], display_labels=k_man_classes)
        disp.plot()
        plt.show()
#proccess_kneigbh()

# plt.plot([32,64,128,224],[0.6243654822335025, 0.6598984771573604, 0.6446700507614214, 0.6446700507614214],label="Euclidian")
# plt.plot([32,64,128,224],[0.5989847715736041, 0.5888324873096447, 0.6091370558375635, 0.6091370558375635],label="Manhattan")
# plt.title("Accuracies")
# plt.xlabel("value of image size")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()
#
# plt.plot([32,64,128,224],[16, 5, 8, 5],label="Euclidian")
# plt.plot([32,64,128,224],[4,2,2,2],label="Manhattan")
# plt.title("Best k for every image size")
# plt.xlabel("image size")
# plt.ylabel("value of k neighboors")
# plt.legend()
# plt.show()