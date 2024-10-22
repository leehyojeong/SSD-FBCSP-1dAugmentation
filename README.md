# SSD+FBCSP+1D-Augmentation

## 진행 상황
- 현재 실험 및 딥러닝 모델 구축 완료
- 국문 논문 작성 완료
- 영문 번역 진행중

## Introduction
- EEG 데이터를 분석하기 위해서는 각각의 세부적인 처리 목적에 따라 특정 주파수의 EEG를 분리해야한다.
- 본 프로젝트에서는 EEG 신호의 종류 중 동작을 상상하는 상황에서 나타나는 EEG 신호를 분석하는 연구에 초점을 맞췄다.
- 최근 Motor Imagery EEG를 분류하기 위해 CNN이나 LSTM, Resnet 등 여러 알고리즘들을 사용하는 방법들이 연구되고 있다. 
- 기존 연구들은 각 사람의 동작을 분류하는 정확도는 높지만 여러 사람의 공통 동작에 대한 분류 정확도는 비교적 높지 않다는 한계점을 보이고 있다.
- 본 프로젝트에서는 기존 연구의 한계점을 개선하여 여러 사람의 공통된 동작을 기존 연구보다 정확하게 분류할 수 있는 SSD+FBCSP+1D-Augmentation 기법을 제안한다.

## System Diagram
![image](https://user-images.githubusercontent.com/39904216/90204835-6b124a00-de1d-11ea-8c19-f8a3f9341afc.png)

## Further Work
- 기존에 존재하는 뇌파 데이터가 아니라 EEG-BCI 기구를 사용하여 직접 측정한 EEG에 이와 동일한 전처리와 알고리즘을 적용해볼 예정이다.
