#from src.data_preparation.data_preparation_original import read_eeg_file
from src.data_preparation.data_preparation(연속 오버래핑) import read_eeg_file
from src.algorithms.csp.CSP import CSP
from scipy import signal
from src.algorithms.fbcsp.MIBIFFeatureSelection import MIBIFFeatureSelection
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
TIME_LENGTH = 2000#2250#750
TIME_WINDOW = 2000#1000#500
EPOCH_SIZE = 2000#None
DATA_FOLDER = "testdata_ssd(0~30)-(10~12)" #"testdata"
K_FOLD = 10
subjects = range(1, 10)
accuracies = {
    "GNB": np.zeros((len(subjects), K_FOLD)),
    "SVM": np.zeros((len(subjects), K_FOLD)),
    "LDA": np.zeros((len(subjects), K_FOLD))
}

total_data=[]
total_label=[]

band_length =2#2#4
min_freq = 2#8#4
max_freq = 30#30#40
bands = [(x, x+band_length) for x in range(min_freq, max_freq, band_length)]
quantity_bands = len(bands)

del band_length
del min_freq
del max_freq

def filter_bank(x):
    data = np.zeros((quantity_bands, *x.shape))
    for n_trial in range(x.shape[0]):
        trial = x[n_trial, :, :]
        filter_bank = np.zeros((quantity_bands, *trial.shape))

        for (i, (low_freq, high_freq)) in enumerate(bands):
            # Create a 5 order Chebyshev Type 2 filter to the specific band (low_freq - high_freq)
            b, a = signal.cheby2(2, 14, [low_freq, high_freq], btype="bandpass", fs=250)
            #b, a = signal.cheby2(5, 48, [low_freq, high_freq], btype="bandpass", fs=250)

            filter_bank[i, :, :] = signal.filtfilt(b, a, trial, axis=0)
        data[:, n_trial, :, :] = filter_bank

    return data

print("Loading data ...")
data_by_subject = []

for subject in subjects:
    left_data_file = f"{DATA_FOLDER}/ssd-{subject}-left.csv" #f"{DATA_FOLDER}/{subject}-left.csv"
    right_data_file = f"{DATA_FOLDER}/ssd-{subject}-right.csv"
    
#    left_data_file = f"{DATA_FOLDER}/{subject}-left.csv" #f"{DATA_FOLDER}/{subject}-left.csv"
#    right_data_file = f"{DATA_FOLDER}/{subject}-right.csv"    right_data_file = f"{DATA_FOLDER}/{subject}-right.csv"
    data = read_eeg_file(left_data_file, right_data_file, TIME_LENGTH, TIME_WINDOW, EPOCH_SIZE)
    

    data.X = np.concatenate((data.left_data, data.right_data))
    data_by_subject.append(data)

del subject
del left_data_file
del right_data_file
del data

print("Data loaded")

for (i, data) in enumerate(data_by_subject):
    print("Subject: ", i+1)

    cv = KFold(n_splits=K_FOLD, shuffle=True)
    for (k, (train_index, test_index)) in enumerate(cv.split(data.X)):
        trials = len(data.left_data)
        print(data.left_data.shape)
        print(data.right_data.shape)

        train_left_index = [index for index in train_index if index < trials]
        train_right_index = [index - trials for index in train_index if index >= trials]
        X_left_train, X_right_train = data.left_data[train_left_index], data.right_data[train_right_index]

        test_left_index = [index for index in test_index if index < trials]
        test_right_index = [index - trials for index in test_index if index >= trials]
        X_left_test, X_right_test = data.left_data[test_left_index], data.right_data[test_right_index]

        y_train, y_test = data.labels[train_index], data.labels[test_index]

        # Feature extraction
        N_CSP_COMPONENTS = 2
        csp_by_band = [CSP(average_trial_covariance=False, n_components=N_CSP_COMPONENTS)
                       for _ in bands]

        left_bands_training = filter_bank(X_left_train)
        right_bands_training = filter_bank(X_right_train)
        left_bands_test = filter_bank(X_left_test)
        right_bands_test = filter_bank(X_right_test)

        features_train = None
        features_test = None
        for n_band in range(quantity_bands):
            left_band_training = left_bands_training[n_band]
            right_band_training = right_bands_training[n_band]
            left_band_test = left_bands_test[n_band]
            right_band_test = right_bands_test[n_band]

            csp = csp_by_band[n_band]
            csp.fit(left_band_training, right_band_training) #왼손오른손 csp적용
            x_train = np.concatenate((left_band_training, right_band_training)) #test, train나눔
            x_test = np.concatenate((left_band_test, right_band_test))
            print("   x_train",x_train.shape)
            if n_band == 0:
                features_train = csp.compute_features(x_train)
                features_test = csp.compute_features(x_test)
            else:
                features_train = np.concatenate((features_train, csp.compute_features(x_train)), axis=1)
                features_test = np.concatenate((features_test, csp.compute_features(x_test)), axis=1)
            #print("   features_train",np.array(features_train).shape)
        

        # Feature Selection
        selected_features = MIBIFFeatureSelection(features_train, features_test, y_train, N_CSP_COMPONENTS, 4, scale=True)

        selected_training_features = selected_features.training_features
        selected_test_features = selected_features.test_features
        '''
        if i==0:
            print("첫번째꺼")
            total_data=selected_training_features
            total_label=y_train 
        else:
            total_data=np.concatenate((total_data,selected_training_features),axis=0)
            total_label=np.concatenate((total_label,y_train),axis=0)
            
        total_data=np.concatenate((total_data,selected_test_features),axis=0)
        total_label=np.concatenate((total_label,y_test),axis=0)
            
        print("결과 ",selected_training_features.shape,total_data.shape,total_label.shape)
        '''
        
        
        np_imgname="images_fbcsp/"
        np_data=np.concatenate((selected_training_features,selected_test_features),axis=0)
        print("features_train", features_train.shape,",N_CSP_COMPONENTS",N_CSP_COMPONENTS,",np_data",np_data.shape)
        np.save(np_imgname+"sub"+str(i+1)+"_"+"data", np_data)
        np.save(np_imgname+"sub"+str(i+1)+"_"+"label",np.concatenate((y_train,y_test),axis=0))
        break #Kfold막는 용도
        
'''
        # GNB classifier
        gnb = GaussianNB()
        gnb.fit(selected_training_features, y_train)
        gnb_predictions = gnb.predict(selected_test_features)
        gnb_accuracy = accuracy_score(y_test, gnb_predictions)
        accuracies["GNB"][i][k] = gnb_accuracy

        # SVM classifier
        svm = SVC(C=.8, kernel="rbf")
        svm.fit(selected_training_features, y_train)
        svm_predictions = svm.predict(selected_test_features)
        svm_accuracy = accuracy_score(y_test, svm_predictions)
        accuracies["SVM"][i][k] = svm_accuracy

        # LDA classifier
        lda = LinearDiscriminantAnalysis()
        lda.fit(selected_training_features, y_train)
        lda_predictions = lda.predict(selected_test_features)
        lda_accuracy = accuracy_score(y_test, lda_predictions)
        accuracies["LDA"][i][k] = lda_accuracy

for classifier in accuracies:
    print(classifier)
    for subject, cv_accuracies in enumerate(accuracies[classifier]):
        acc_mean = np.mean(cv_accuracies)
        acc_std = np.std(cv_accuracies)
        print(f"\tSubject {subject+1} average accuracy: {acc_mean:.4f} +/- {acc_std:.4f}")
    average_acc_mean = np.mean(accuracies[classifier])
    average_acc_std = np.std(accuracies[classifier])
    print(f"\tAverage accuracy: {average_acc_mean:.4f} +/- {average_acc_std:.4f}")
'''

#np_imgname="images_fbcsp/"
#np.save(np_imgname+"data_SSDFB_len20002000", total_data)
#np.save(np_imgname+"label_SSDFB_len20002000",total_label)
