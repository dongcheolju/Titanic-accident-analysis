# 필요한 라이브러리 불러오기
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 한글 사용을 위한 폰트 불러오기
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
A = "data/titanic.csv"
titanic_data = pd.read_csv(A, encoding='cp949') # 한국어를 불러올 수 있게 인코딩 지정

# 데이터 전처리
titanic_data = titanic_data[['객실 등급', '성별', '나이', '형제자매/배우자 수', '부모/자식 수 ', '요금', '생존']]
titanic_data = titanic_data.dropna(subset=['나이', '형제자매/배우자 수', '부모/자식 수 '])  # 누락된 값이 있는 행 삭제

# 성별을 숫자로 매핑
titanic_data['성별'] = titanic_data['성별'].map({'male': 0, 'female': 1})

# Features와 Labels 정의
X = titanic_data.drop('생존', axis=1)
y = titanic_data['생존']

# 학습 데이터와 테스트 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 생성
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 모델 학습
model.fit(X_train, y_train)

# 테스트 데이터로 예측
y_pred = model.predict(X_test)

# 정확도 출력
accuracy = accuracy_score(y_test, y_pred)
print("정확도:", accuracy)

# 분류 보고서 출력
print("\n분류 보고서:\n", classification_report(y_test, y_pred))

# 각 요소의 중요도 시각화
feature_importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_importances)), feature_importances, align="center")
plt.yticks(range(len(feature_importances)), feature_names)
plt.xlabel("특성 중요도")
plt.title("생존 여부의 특성 중요도")
plt.savefig("titanic_analysis_plots/생존 여부의 특성 중요도")
plt.show()

# 나이에 따른 생존 여부 시각화
plt.figure(figsize=(12, 6))
sns.histplot(x='나이', hue='생존', data=titanic_data, kde=True, multiple='stack', bins=30, palette='husl')
plt.title('나이에 따른 생존 여부')
plt.savefig("titanic_analysis_plots/나이에 따른 생존 여부")
plt.show()

# 성별에 따른 생존 여부 시각화
# 성별 매핑을 다시 원래 값으로 변경
titanic_data['성별'] = titanic_data['성별'].map({0: 'male', 1: 'female'})
plt.figure(figsize=(8, 5))
sns.countplot(x='성별', hue='생존', data=titanic_data, palette='husl')
plt.title('성별에 따른 생존 여부')
plt.savefig("titanic_analysis_plots/성별에 따른 생존 여부")
plt.show()

# 요금에 따른 생존 여부 시각화
plt.figure(figsize=(12, 6))
sns.histplot(x='요금', hue='생존', data=titanic_data, kde=True, multiple='stack', bins=30, palette='husl')
plt.title('요금에 따른 생존 여부')
plt.savefig("titanic_analysis_plots/요금에 따른 생존 여부")
plt.show()

# 요금 구간별 사망자 비율 시각화
# 요금을 구간별로 나누기
bins = [0, 50, 100, 150, 200, 250, 300, 350, float('inf')]
labels = ['0-50', '51-100', '101-150', '151-200', '201-250', '251-300', '301-350', '350+']
titanic_data['요금 구간'] = pd.cut(titanic_data['요금'], bins=bins, labels=labels, include_lowest=True)
# 요금별 사망자와 생존자의 수 계산
fare_survival_counts = titanic_data.groupby(['요금 구간', '생존'], observed=False).size().unstack()
# 비율 계산
fare_survival_ratio = fare_survival_counts.div(fare_survival_counts.sum(axis=1), axis=0)
# 그래프 그리기
fig, ax = plt.subplots(figsize=(12, 8))
fare_survival_ratio.plot(kind='bar', stacked=False, color=['lightcoral', 'skyblue'], ax=ax)
ax.set_title('요금별 사망자와 생존자 비율')
ax.set_xlabel('요금 구간')
ax.set_ylabel('비율')
ax.legend(title='생존 여부', labels=['사망', '생존'])
plt.savefig("titanic_analysis_plots/요금구간에 따른 생존 비율")
plt.show()
