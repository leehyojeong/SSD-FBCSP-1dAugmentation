from src.data_preparation.data_preparation import read_eeg_file
from scipy import signal
from src.algorithms.csp.CSP import CSP
import pywt
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import numpy as np

TIME_LENGTH = 750
TIME_WINDOW = 500
EPOCH_SIZE = None
DATA_FOLDER = "testdata"
CSP_COMPONENTS = 2
FS = 250
WAVELET = "coif1"
K_FOLD = 10
subjects = range(1, 10)
subjects_set = set(subjects)
accuracies = {
    "GNB": np.zeros((len(subjects), K_FOLD)),
    "SVM": np.zeros((len(subjects), K_FOLD)),
    "LDA": np.zeros((len(subjects), K_FOLD))
}

for subject in subjects:
    print("========= Subject: ", subject)
    # Load data
    print("Loading data ...")
    left_data_file = f"{DATA_FOLDER}/{subject}-left.csv"
    right_data_file = f"{DATA_FOLDER}/{subject}-right.csv"
    data = read_eeg_file(left_data_file, right_data_file, TIME_LENGTH, TIME_WINDOW, EPOCH_SIZE)

    # Pre-processing
    print("Pre-processing ...")
    print("Applying 5º order Butterworth bandpass filter (7-30 Hz)")
    b, a = signal.butter(5, [7, 30], btype="bandpass", fs=FS)

    data.left_data = signal.filtfilt(b, a, data.left_data, axis=1)
    data.right_data = signal.filtfilt(b, a, data.right_data, axis=1)

    print("Spatial-filtering ...")
    data.X = np.concatenate((data.left_data, data.right_data))

    csp = CSP(average_trial_covariance=True, n_components=CSP_COMPONENTS)
    csp.fit(data.left_data, data.right_data)
    data.Z = np.array([csp.project(x) for x in data.X])

    # Feature extraction
    print("Extracting features ...")
    data.F = np.zeros((data.X.shape[0], 2, CSP_COMPONENTS))
    for n_epoch in range(0, data.X.shape[0]):
        epoch = data.Z[n_epoch]

        # Calculate the wavelet features
        for n_feature in range(0, data.F.shape[2]):
            alpha_band, beta_band = pywt.dwt(epoch[:, n_feature], WAVELET)
            data.F[n_epoch, 0, n_feature] = np.sum(beta_band ** 2)

        # Calculate the frequency-domain features
        psd_window_size = 100
        psd_window_overlap = psd_window_size//2
        beta_freqs = range(13, 31)
        for n_feature in range(0, data.F.shape[2]):
            freq, psd = signal.welch(epoch[:, n_feature], fs=FS, window="hanning",
                                     nperseg=psd_window_size, noverlap=psd_window_overlap)
            data.F[n_epoch, 1, n_feature] = np.sum(psd[beta_freqs] ** 2)

    len_features = data.F.shape[1] * data.F.shape[2]
    data.F = np.reshape(data.F, newshape=(data.F.shape[0], len_features))

    # Feature normalization
    data.F = stats.zscore(data.F, axis=0)

    # Classification
    print("Classifying features ...")
    subject_index = subject - 1

    cv = KFold(n_splits=K_FOLD, shuffle=True)
    for (k, (train_index, test_index)) in enumerate(cv.split(data.F)):
        X_train, X_test = data.F[train_index], data.F[test_index]
        y_train, y_test = data.labels[train_index], data.labels[test_index]

        # GNB classifier
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        gnb_predictions = gnb.predict(X_test)
        gnb_accuracy = accuracy_score(y_test, gnb_predictions)
        print(f"GNB accuracy: {gnb_accuracy:.4f}")
        accuracies["GNB"][subject_index][k] = gnb_accuracy

        # SVM classifier
        svm = SVC(C=.8, gamma="scale", kernel="rbf")
        svm.fit(X_train, y_train)
        svm_predictions = svm.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_predictions)
        print(f"SVM accuracy: {svm_accuracy:.4f}")
        accuracies["SVM"][subject_index][k] = svm_accuracy

        # LDA classifier
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        lda_predictions = lda.predict(X_test)
        lda_accuracy = accuracy_score(y_test, lda_predictions)
        print(f"LDA accuracy: {lda_accuracy:.4f}")
        accuracies["LDA"][subject_index][k] = lda_accuracy

    print()

for classifier in accuracies:
    print(classifier)
    for subject, cv_accuracies in enumerate(accuracies[classifier]):
        acc_mean = np.mean(cv_accuracies)
        acc_std = np.std(cv_accuracies)
        print(f"\tSubject {subject+1} average accuracy: {acc_mean:.4f} +/- {acc_std:.4f}")
    average_acc_mean = np.mean(accuracies[classifier])
    average_acc_std = np.std(accuracies[classifier])
    print(f"\tAverage accuracy: {average_acc_mean:.4f} +/- {average_acc_std:.4f}")