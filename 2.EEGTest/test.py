import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, find_events, pick_types, set_eeg_reference
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from mne import viz
import os
import csv

#======================================================================================================
# 테스트 : 파일하나 읽고, rejected trial 제거하고, epochs까지 구하는 기능 테스트 완료
#======================================================================================================
def read_single_file(filename):
    raw = mne.io.read_raw_gdf(filename)

    # band-pass filter 적용하기
    #raw.filter(7., 35., fir_design='firwin', skip_by_annotation='edge')

    events_A, event_id_A = mne.events_from_annotations(raw);
    print(events_A,event_id_A)
    # rejected 되어야 하는 event 찾기
    myrejectid = np.where(events_A[:, 2] == event_id_A['1023'])
    myjejectedevent = []
    for i in myrejectid:
        myjejectedevent.append(i+1)
    events = []

    # # rejected trials 제거
    for j in range(0, events_A.shape[0]):
        if j in myjejectedevent[0]:
            print("제거됨")
        else:
            events.append(events_A[j])
    events = np.asarray(events)
    #
    picks = mne.pick_channels(raw.info["ch_names"], ["EEG:C3", "EEG:Cz", "EEG:C4"])
    #
    # #큐 사인 이후 0.5에서 3.5초 데이터 사용
    tmin, tmax = 0.5, 3.5
    event_ids = dict(handsleft=event_id_A['769'], handsright=event_id_A['770'])
    epochs = mne.Epochs(raw, events, event_ids, tmin, tmax, picks=picks, baseline=None, preload=True)
    data = epochs['handsleft'].get_data()[0].T
    data = data[:750][:]
    print(data.shape)
    # print('Data : {}'.format(epochs['handsleft'].get_data()))
    # print('Data : {}'.format(epochs['handsright'].get_data()))

    # fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15, 10))
    # ax1.plot(epochs['handsleft'].get_data()[:,1,:][0]);
    # ax2.plot(epochs['handsright'].get_data()[:,1,:][0]);
    # plt.legend(raw.ch_names[:3], loc=1);
    # plt.show()

    # 각 이벤트에 대한 데이터 그리기
    # print(epochs['handsright'].get_data().shape)
    # plt.plot(epochs['handsright'].get_data()[:,1,:][0])
    # plt.show()

    #이벤트가 발생한 곳 그림 그리기
    # df=pd.DataFrame(events_A, columns = ['time','x','event'])
    # df.index = df['time']/1024
    # fig, ax = plt.subplots()
    # ax.scatter(df.index, df['event'])
    # indexA = np.array(df.index[df['event']==10])
    # print(indexA)
    # for i in indexA:
    #     ax.axvline(x=i, ymin=0, ymax=1, c='r', linestyle='--')
    # plt.show()
#======================================================================================================

#======================================================================================================
# subject단위로 전체 디렉토리 읽어서 왼손 / 오른손 데이터들을 통합하여 파일에 저장하기 위한 기능
#  파일 이름 : 예) 1-left.csv, l-right.csv 형태로 저장하며, 1은 1번 참여자를 의미함
#======================================================================================================
def read_gdf(filenames, subject, mint, maxt):
    f1 = open("testdata/"+subject+"-left.csv",'w', newline='')
    f2 = open("testdata/"+subject+"-right.csv",'w', newline='')
    wleft = csv.writer(f1)
    wright = csv.writer(f2)
    lefthands = []
    righthands = []
    for filename in filenames :
        raw = mne.io.read_raw_gdf(filename)
        events_A, event_id_A = mne.events_from_annotations(raw);
        #print(event_id_A)
        # rejected 되어야 하는 event 찾기
        myrejectid = np.where(events_A[:, 2] == event_id_A['1023'])
        myjejectedevent = []
        for i in myrejectid:
            myjejectedevent.append(i+1)

        # # rejected trials 제거
        events = []
        for j in range(0, events_A.shape[0]):
            if j in myjejectedevent[0]:
                print("제거됨")
            else:
                events.append(events_A[j])
        events = np.asarray(events)
        picks = mne.pick_channels(raw.info["ch_names"], ["EEG:C3", "EEG:Cz", "EEG:C4"])

        # #큐 사인 이후 0.5에서  3.5초 데이터 저장
        tmin = mint
        tmax = maxt
        event_ids = dict(handsleft=event_id_A['769'], handsright=event_id_A['770'])
        epochs = mne.Epochs(raw, events, event_ids, tmin, tmax, picks=picks, baseline=None, preload=True)

        # 파일에 저장하기 // 3초 이므로 250*3 ==> 750개 씩 저장
        for i in range(0, epochs['handsleft'].get_data().shape[0]) :
            data = epochs['handsleft'].get_data()[i].T
            count =0;
            for j in data:
                if count>=750 :
                    break;
                else:
                    count = count+1
                wleft.writerow(np.asarray(j))

        for i in range(0, epochs['handsright'].get_data().shape[0]):
            data = epochs['handsright'].get_data()[i].T
            count = 0;
            for j in data:
                if count >= 750:
                    break;
                else:
                    count = count + 1
                wright.writerow(np.asarray(np.asarray(j)))
    f1.close()
    f2.close()
#======================================================================================================

#********************************* 테스트 부분 ********************************************************

#======================================================================================================
# 하나의 파일 읽어서 테스트하기
#======================================================================================================
#read_single_file("gdf/B0101T.gdf")

#======================================================================================================
# 전체 디렉토리 읽어서 파일 합치기 ==> subject별 csv 파일 만들기
# 큐 사인후 0.5부터, 3초 동안의 데이터 저장하기
#======================================================================================================
#os.makedirs("./testdata")
subjects = range(1, 10)
subjects_set = set(subjects)
for subject in subjects:
    myPath = f"gdf/{subject}"
    raw_fnames = [os.path.join(myPath,f) for f in os.listdir(myPath)]
    read_gdf(raw_fnames, f"{subject}", 0.5, 3.5)



