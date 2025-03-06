import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import numpy as np

# 读取训练集数据
train_data = pd.read_csv('train_data.csv')

# 分离输入特征和目标变量
X = train_data[['Age', 'Sex', 'Race', 'Hisogical.Type',
                'T', 'Liver.metastasis', 'Radiation',
                'Chemotherapy', 'Marital.status']]
y = train_data['Vital.status']

# 处理缺失值和无穷大值
def handle_non_finite(X, y):
    # 合并 X 和 y 以便同时处理索引
    combined = pd.concat([X, y], axis=1)
    # 检查是否存在缺失值
    if combined.isnull().any().any():
        combined = combined.dropna()
    # 检查是否存在无穷大值
    if np.isinf(combined).any().any():
        combined = combined.replace([np.inf, -np.inf], np.nan).dropna()
    # 分离处理后的 X 和 y
    X = combined.drop('Vital.status', axis=1)
    y = combined['Vital.status']
    return X, y

X, y = handle_non_finite(X, y)

# 特征映射
class_mapping = {0: "Alive", 1: "Dead"}
Age_mapper = {'＜60': 1, '60-73': 2, '＞73': 3}
Sex_mapper = {'male': 1, 'female': 2}
Race_mapper = {"White": 1, "Black": 2, "Other": 3}
Hisogical_Type_mapper = {"Adenocarcinoma": 1, "Squamous-cell carcinoma": 2}
T_mapper = {"T4": 4, "T1": 1, "T2": 2, "T3": 3}
Liver_metastasis_mapper = {"NO": 1, "Yes": 2}
Radiation_mapper = {"NO": 1, "Yes": 2}
Chemotherapy_mapper = {"NO": 1, "Yes": 2}
Marital_status_mapper = {"Married/Partnered": 1, "Unmarried/Unstable Relationship": 2}

# 对训练数据进行特征映射
X['Age'] = X['Age'].map(Age_mapper)
X['Sex'] = X['Sex'].map(Sex_mapper)
X['Race'] = X['Race'].map(Race_mapper)
X['Hisogical.Type'] = X['Hisogical.Type'].map(Hisogical_Type_mapper)
X['T'] = X['T'].map(T_mapper)
X['Liver.metastasis'] = X['Liver.metastasis'].map(Liver_metastasis_mapper)
X['Radiation'] = X['Radiation'].map(Radiation_mapper)
X['Chemotherapy'] = X['Chemotherapy'].map(Chemotherapy_mapper)
X['Marital.status'] = X['Marital.status'].map(Marital_status_mapper)

# 再次处理映射后可能出现的非有限值
X, y = handle_non_finite(X, y)

# 检查数据类型
X = X.astype('float64')
y = y.astype('float64').squeeze()

# 创建并训练Random Forest模型
rf_params = {
    'n_estimators': 200,
    'min_samples_split': 10,
    'max_depth': 20
}

rf_model = RandomForestClassifier(**rf_params)
rf_model.fit(X, y)

# 预测函数
def predict_Vital_status(age, sex, race, histologic_type,
                         t, liver_metastasis, radiation,
                         chemotherapy, marital_status):
    input_data = pd.DataFrame({
        'Age': [Age_mapper[age]],
        'Sex': [Sex_mapper[sex]],
        'Race': [Race_mapper[race]],
        'Hisogical.Type': [Hisogical_Type_mapper[histologic_type]],
        'T': [T_mapper[t]],
        'Liver.metastasis': [Liver_metastasis_mapper[liver_metastasis]],
        'Radiation': [Radiation_mapper[radiation]],
        'Chemotherapy': [Chemotherapy_mapper[chemotherapy]],
        'Marital.status': [Marital_status_mapper[marital_status]]
    })
    prediction = rf_model.predict(input_data)[0]
    probability = rf_model.predict_proba(input_data)[0][1]  # 获取属于类别1的概率
    class_label = class_mapping[prediction]
    return class_label, probability

# 创建Web应用程序
st.title("6 - month survival of ECLM patients based on Random Forest")
st.sidebar.write("Variables")

age = st.sidebar.selectbox("Age", options=list(Age_mapper.keys()))
sex = st.sidebar.selectbox("Sex", options=list(Sex_mapper.keys()))
race = st.sidebar.selectbox("Race", options=list(Race_mapper.keys()))
histologic_type = st.sidebar.selectbox("Histologic Type", options=list(Hisogical_Type_mapper.keys()))
t = st.sidebar.selectbox("T", options=list(T_mapper.keys()))
liver_metastasis = st.sidebar.selectbox("Liver metastasis", options=list(Liver_metastasis_mapper.keys()))
radiation = st.sidebar.selectbox("Radiation", options=list(Radiation_mapper.keys()))
chemotherapy = st.sidebar.selectbox("Chemotherapy", options=list(Chemotherapy_mapper.keys()))
marital_status = st.sidebar.selectbox("Marital status", options=list(Marital_status_mapper.keys()))

if st.button("Predict"):
    prediction, probability = predict_Vital_status(
        age, sex, race, histologic_type, t, liver_metastasis, radiation,
        chemotherapy, marital_status
    )

    st.write("Predicted Vital Status:", prediction)
    st.write("Probability of 6 - month survival is:", 1 - probability)
