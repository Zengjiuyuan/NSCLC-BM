# 6个月.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# 读取训练集数据
train_data = pd.read_csv('train_data.csv')

# 分离输入特征和目标变量
X = train_data[['Age', 'Sex', 'Race', 'Hisogical.Type',
                'T', 'Liver.metastasis', 'Radiation',
                'Chemotherapy', 'Marital.status']]
y = train_data['Vital.status']

# 创建并训练随机森林模型
rf_params = {
    'n_estimators': 200,
    'min_samples_split': 10,
    'max_depth': 20,
    'random_state': 42
}

rf_model = RandomForestClassifier(**rf_params)
rf_model.fit(X, y)

# 特征映射
class_mapping = {0: "Alive", 1: "Dead"}
Age_mapper = {'＜60': 1, '60-73': 2, '＞73': 3}
Sex_mapper = {'male': 1, 'female': 2}
Race_mapper = {"White": 1, "Black": 2, "Other": 3}
Hisogical.Type_mapper = {"Adenocarcinoma": 1, "Squamous-cell carcinoma": 2}
T_mapper = {"T4": 4, "T1": 1, "T2": 2, "T3": 3}
Liver.metastasis_mapper = {"NO": 1, "Yes": 2}
Radiation_mapper = {"NO": 1, "Yes": 2}
Chemotherapy_mapper = {"NO": 1, "Yes": 2}
Marital.status_mapper = {"Married/Partnered": 1, "Unmarried/Unstable Relationship": 2}

# 预测函数
def predict_Vital_status(age, sex, race, hisogical_type, t, 
                        liver_metastasis, radiation, chemotherapy, 
                        marital_status):
    input_data = pd.DataFrame({
        'Age': [Age_mapper[age]],
        'Sex': [Sex_mapper[sex]],
        'Race': [Race_mapper[race]],
        'Hisogical.Type': [Hisogical.Type_mapper[hisogical_type]],
        'T': [T_mapper[t]],
        'Liver.metastasis': [Liver.metastasis_mapper[liver_metastasis]],
        'Radiation': [Radiation_mapper[radiation]],
        'Chemotherapy': [Chemotherapy_mapper[chemotherapy]],
        'Marital.status': [Marital.status_mapper[marital_status]]
    })
    prediction = rf_model.predict(input_data)[0]
    probability = rf_model.predict_proba(input_data)[0][1]
    return class_mapping[prediction], probability

# 创建Web应用程序
st.title("6-month survival of ECLM patients based on Random Forest")
st.sidebar.write("Variables")

age = st.sidebar.selectbox("Age", options=list(Age_mapper.keys()))
sex = st.sidebar.selectbox("Sex", options=list(Sex_mapper.keys()))
race = st.sidebar.selectbox("Race", options=list(Race_mapper.keys()))
hisogical_type = st.sidebar.selectbox("Hisogical Type", options=list(Hisogical.Type_mapper.keys()))
t = st.sidebar.selectbox("T stage", options=list(T_mapper.keys()))
liver_metastasis = st.sidebar.selectbox("Liver metastasis", options=list(Liver.metastasis_mapper.keys()))
radiation = st.sidebar.selectbox("Radiation", options=list(Radiation_mapper.keys()))
chemotherapy = st.sidebar.selectbox("Chemotherapy", options=list(Chemotherapy_mapper.keys()))
marital_status = st.sidebar.selectbox("Marital status", options=list(Marital.status_mapper.keys()))

if st.button("Predict"):
    prediction, probability = predict_Vital_status(
        age, sex, race, hisogical_type, t,
        liver_metastasis, radiation, chemotherapy,
        marital_status
    )
    
    st.write("Predicted Vital Status:", prediction)
    st.write(f"Probability of 6-month survival is: {probability:.2%}")
