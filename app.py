import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 초기 설정
st.set_page_config(page_title="Sleep Disorder Dashboard", layout="wide")

# 데이터 불러오기
df = pd.read_csv("data/Health_Sleep_Statistics.csv")
df.drop(columns=["Bedtime", "Wake-up Time", "User ID"], inplace=True, errors='ignore')

# 결측치 제거
df.dropna(inplace=True)
# 레이블 인코딩할 컬럼 지정
label_cols = df.select_dtypes(include=['object']).columns

# 라벨 인코딩 적용
from sklearn.preprocessing import LabelEncoder
le_dict = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# 레이블 인코딩
label_cols = ['Gender', 'Dietary Habits', 'Sleep Disorders']
le_dict = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# 특성과 타겟 분리
X = df.drop(columns=["Sleep Disorders"])
y = df["Sleep Disorders"]

# 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
conf_matrix = confusion_matrix(y_test, y_pred)

# 특성 중요도
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

# ================= Streamlit UI =================== #
st.sidebar.title("💤 수면장애 분석")
menu = st.sidebar.selectbox("탭 선택", ["Home", "데이터 분석", "데이터 시각화", "머신러닝 보고서"])


# Home
if menu == "Home":
    st.title("Sleep Disorder 분석 대시보드")
    st.markdown("""
    - 본 대시보드는 **건강 및 수면 통계 데이터셋**을 활용하여 다양한 라이프스타일 요인을 바탕으로 **수면 장애 여부**를 분석하고 분류하는 것을 목표로 합니다.
    - **타겟 컬럼** : Sleep Disorders (0: 없음, 1: 있음)
    """, unsafe_allow_html=True)

    st.subheader("Health Sleep Statistics")
    st.dataframe(df.head(10))

# 데이터 분석
elif menu == "데이터 분석":
    st.title("데이터 분석")
    tab1, tab2 = st.tabs(["기술 통계", "조건 검색"])
    
    with tab1:
        st.subheader("통계 요약")
        st.dataframe(df.describe())

    with tab2:
        st.subheader("조건별 필터")
        column = st.selectbox("컬럼 선택", df.columns)
        value = st.selectbox("값 선택", df[column].unique())
        filtered = df[df[column] == value]
        st.write(f"총 {filtered.shape[0]}건의 데이터")
        st.dataframe(filtered)

# 데이터 시각화
elif menu == "데이터 시각화":
    st.title("데이터 시각화")
    tab1, tab2, tab3 = st.tabs(["히스토그램", "박스플롯", "히트맵"])

    with tab1:
        st.subheader("히스토그램")
        col = st.selectbox("변수 선택", df.drop(columns=["Sleep Disorders"]).columns)
        fig1, ax1 = plt.subplots()
        sns.histplot(data=df, x=col, kde=True, hue="Sleep Disorders", ax=ax1)
        st.pyplot(fig1)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Gender vs Sleep Quality vs Sleep Disorders (Boxplot)")
            fig2, ax2 = plt.subplots(figsize=(3.5, 2.5))
            sns.boxplot(data=df, x="Gender", y="Sleep Quality", hue="Sleep Disorders", ax=ax2)
            st.pyplot(fig2)
        with col2:
            st.subheader("Gender vs Calories Burned vs Sleep Disorders (Boxplot)")
            fig3, ax3 = plt.subplots(figsize=(3.5, 2.5))
            sns.boxplot(data=df, x="Gender", y="Calories Burned", hue="Sleep Disorders", ax=ax3)
            st.pyplot(fig3)

    with tab3:
        st.subheader("상관관계 히트맵")
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax4)
        st.pyplot(fig4)

# 머신러닝 보고서
elif menu == "머신러닝 보고서":
    st.title("머신러닝 모델 성능")
    st.write(f"### 정확도 (Accuracy): `{accuracy:.2f}`")
    
    st.subheader("분류 리포트")
    st.dataframe(report_df.round(2))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        fig6, ax6 = plt.subplots(figsize=(2.5, 1.5))

        # heatmap 그리기
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax6, annot_kws={"size": 6})  # 셀 내 글자 크기 설정

        # 축 레이블과 숫자 크기 설정
        ax6.tick_params(axis='both', labelsize=5)  # x축과 y축 레이블 글자 크기 6

        # 색상 막대 글자 크기 설정
        colorbar = ax6.collections[0].colorbar  # 색상 막대 객체 가져오기
        colorbar.ax.tick_params(labelsize=5)  # 색상 막대 글자 크기 설정

        st.pyplot(fig6)

    with col2:
        st.subheader("특성 중요도")
        fig6, ax6 = plt.subplots(figsize=(5, 4))
        sns.barplot(x="Importance", y="Feature", data=feature_importance, ax=ax6)
        st.pyplot(fig6)
