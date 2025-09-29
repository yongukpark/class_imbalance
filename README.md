# Hop 기반 가중 앙상블을 통한 클래스 불균형 문제 해결 (Hop-based Weighted Ensemble for Class Imbalance Classification)

이 프로젝트는 **클래스 불균형** 환경에서 소수 클래스(minority class)의 탐지 성능을 높이기 위한 다양한 기법들을 구현하고 비교하는 실험 코드입니다.  
제안 방법인 **Hop 기반 가중 앙상블(Hop-based Weighted Ensemble)** 을 포함하여, 전통적인 오버샘플링/언더샘플링 및 클래스 가중치 기반 학습 기법과 비교합니다.


## 프로젝트 구조
```
class_imbalance/
│
├── src/
│ ├── experiment.py # Hop 기반 제안 기법(Proposed) 실행
│ ├── experiment_2.py # 기존의 Sampling 기법 실행
│ ├── analysis.py # Proposed 방식과 기존의 방식 비교
│ ├── utils.py # 공통 유틸 함수
│ ├── dataset.py # 데이터 로드/전처리
│ └── metrics.py # 성능 평가 함수
│
├── results/ # 결과 저장 폴더
├── requirements.txt # 필요한 패키지 목록
└── README.md
```

## 환경 설정

### 1. Conda 환경 생성
```bash
conda create -n imbalance python=3.9
conda activate imbalance
```
### 2. 필수 라이브러리 설치
```bash
pip install -r requirements.txt
```
##  실행 방법
### 1. Hop 기반 앙상블 실행 (제안기법)
```bash
python src/experiment.py
```
결과는 ./reslts/에 csv 형태로 저장됨

### 2. 베이스라인 및 기존 기법들 실행
```bash
python src/experiment_2.py
```
결과는 ./reslts/에 csv 형태로 저장됨
### 3. 결과 분석
```bash
python src/compare.py
```
생성된 csv를 토대로 제안기법과 기존기법의 성능 비교


### 기타
**사용 기법**
* Proposed: Hop-based Weighted Ensemble

* Baseline: XGBoost

* Oversampling: RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE

* Undersampling: RandomUnderSampler

* Hybrid: SMOTETomek, SMOTEENN

* Ensemble: EasyEnsembleClassifier

* Loss-based: Class Balanced Loss (scale_pos_weight)