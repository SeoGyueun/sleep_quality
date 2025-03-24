import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===== 데이터 불러오기 및 전처리 =====
df = pd.read_csv("data/Health_Sleep_Statistics.csv")

df = df.drop(columns=['User ID'])  # 불필요한 컬럼 제거

# 결측치 처리 (NaN 값 제거)
df = df.dropna()

# 범주형 변수 인코딩
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

le_physical_activity_level = LabelEncoder()
df['Physical Activity Level'] = le_physical_activity_level.fit_transform(df['Physical Activity Level'])

le_dietary_habits = LabelEncoder()  
df['Dietary Habits'] = le_dietary_habits.fit_transform(df['Dietary Habits']) 

# 'Sleep Disorders' 및 'Medication Usage' 값 처리
df['Sleep Disorders'] = df['Sleep Disorders'].map({'no': 0, 'yes': 1})
df['Medication Usage'] = df['Medication Usage'].map({'no': 0, 'yes': 1})

# Sleep Quality를 인코딩 (타겟 변수)
le_sleep_quality = LabelEncoder()
df['Sleep Quality'] = le_sleep_quality.fit_transform(df['Sleep Quality'])

# 'Sleep Duration' 계산
def convert_time(bedtime, wakeup):
    bedtime = pd.to_datetime(bedtime, format='%H:%M')
    wakeup = pd.to_datetime(wakeup, format='%H:%M')
    sleep_duration = (wakeup - bedtime).dt.total_seconds() / 3600
    sleep_duration[sleep_duration < 0] += 24  # 음수 값 보정
    return sleep_duration

df['Sleep Duration'] = convert_time(df['Bedtime'], df['Wake-up Time'])
df = df.drop(columns=['Bedtime', 'Wake-up Time'])

# 특성과 타겟 분리
X = df.drop(columns=['Sleep Quality'])
y = df['Sleep Quality']

# 'Daily Steps', 'Calories Burned' 등 범주형 변수 처리
categorical_columns = ['Daily Steps', 'Calories Burned']  # 여기에 범주형 변수 추가

for col in categorical_columns:
    if df[col].dtype == 'object':  # 만약 컬럼에 문자열이 포함된 경우
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 훈련/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 성능 평가 함수
def classification_report_to_df(report):
    df_report = pd.DataFrame(report).transpose()
    return df_report

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 문제를 해결하려면, labels 파라미터를 사용하여 예측된 클래스만 넘기도록 수정
labels = le_sleep_quality.classes_  # 실제 클래스명만 사용
report_dict = classification_report(y_test, y_pred, labels=np.unique(y), target_names=labels.tolist(), output_dict=True)

classification_df = classification_report_to_df(report_dict)

# ===== Streamlit UI 구성 =====
st.set_page_config(page_title='Health & Sleep Dashboard', layout='wide')
st.sidebar.title('건강과 수면 분석')
menu = st.sidebar.selectbox('Menu', ['Home', 'EDA', 'Model Performance'])

def home():
    st.title('Health & Sleep Statics')
    st.text(''' 
        다양한 요소에 따라 달라지는 수면의 질을 나타낸 통계입니다 
    ''')
    st.markdown('''  
    - **Sleep Quality**: 수면의 질 (타겟 변수)
    - **Age**: 나이
    - **Gender**: 성별 (0: Female, 1: Male)
    - **Sleep Duration**: 수면 시간 (시간 단위)
    - **Daily Steps**: 하루 걸음 수
    - **Calories Burned**: 소모 칼로리
    - **Physical Activity Level**: 신체 활동 수준
    - **Dietary Habits**: 식습관
    - **Sleep Disorders**: 수면 장애 여부 (0: No, 1: Yes)
    - **Medication Usage**: 약물 복용 여부 (0: No, 1: Yes)
    ''')
    
    # CSV 파일 불러오기
    data = pd.read_csv('data/Health_Sleep_Statistics.csv')
    
    # 데이터 상위 10개 행 보여주기
    st.dataframe(data.head(10))

def eda():
    st.title('데이터 시각화')
    chart_tabs = st.tabs(['Histogram', 'Boxplot', 'Heatmap'])
    
    with chart_tabs[0]:
        st.subheader('Feature Distributions')
        fig, axes = plt.subplots(2,2, figsize=(15,10))
        columns = ['Age', 'Daily Steps', 'Calories Burned', 'Sleep Duration']
        for i, col in enumerate(columns):
            ax = axes[i//2, i%2]
            sns.histplot(df[col], bins=20, kde=True, ax=ax)
            ax.set_title(col)
        st.pyplot(fig)
    
    with chart_tabs[1]:
        st.subheader('Boxplot: Sleep Quality by Sleep Disorders and Sleep Duration')
        
        # 데이터 변형: 여러 변수를 x축에 넣기 위해 melt() 사용
        df_melted = df.melt(id_vars=['Sleep Quality'], value_vars=['Sleep Disorders', 'Sleep Duration'],
                            var_name='Category', value_name='Value')
        
        # Boxplot 생성
        fig, ax = plt.subplots(figsize=(8,5))
        sns.boxplot(data=df_melted, x='Value', y='Sleep Quality', hue='Category', palette='Set2', ax=ax)
        
        st.pyplot(fig)
    
    with chart_tabs[2]:
        st.subheader('Feature Correlation Heatmap')
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
        st.pyplot(fig)

def model_performance():
    st.title('모델 성능 평가')
    st.write(f'**Accuracy:** {accuracy:.2f}')
    st.text('Classification Report:')
    st.dataframe(classification_df)

if menu == 'Home':
    home()
elif menu == 'EDA':
    eda()
elif menu == 'Model Performance':
    model_performance()
