# -*- coding: utf-8 -*-
"""
streamlit_app.py — แบบฟอร์มทำนาย "อาชีพที่เหมาะสม" จากผลการเรียน
การใช้งาน:
    1) วางไฟล์โมเดล 'career_fit_model.pkl' ไว้โฟลเดอร์เดียวกับไฟล์นี้
       หรือไว้ในโฟลเดอร์ 'artifacts/' (ชื่อไฟล์เหมือนเดิม)
    2) รันคำสั่ง:  streamlit run streamlit_app.py
"""

from pathlib import Path
import sklearn
import pickle
import sklearn
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Career Recommender", layout="wide")

# ---------- Load model ----------
@st.cache_resource
def load_bundle():
    candidates = [
        Path("career_fit_model.pkl"),
        
    ]
    for p in candidates:
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
    st.error("ไม่พบไฟล์โมเดล 'career_fit_model.pkl' กรุณาวางไฟล์ไว้โฟลเดอร์เดียวกัน หรือในโฟลเดอร์ 'artifacts/'")
    st.stop()

bundle = load_bundle()
model = bundle["model"]
feature_cols = bundle["feature_cols"]
grade_to_points = bundle.get("grade_to_points", {"A":4.0, "B+":3.5, "B":3.0, "C+":2.5, "C":2.0, "D+":1.5, "D":1.0, "F":0.0})
subject_cols = [c for c in feature_cols if c not in ["เพศ", "ชั้นปี", "GPA", "เพศ_หญิง", "เพศ_ชาย"]]

# ---------- UI ----------
st.title("🔎 ทำนายอาชีพที่เหมาะสมด้วย Data Science")
st.caption("กรอกข้อมูลผลการเรียนและรายละเอียด แล้วกดปุ่มเพื่อทำนายอาชีพที่เหมาะสม")

col_left, col_right = st.columns([1,1])

with col_left:
    gender = st.selectbox("เพศ", ["ชาย", "หญิง"])
    year = st.selectbox("ชั้นปี", [1,2,3,4], index=3)
    gpa = st.number_input("GPA", min_value=0.00, max_value=4.00, value=3.20, step=0.01, format="%.2f")

grade_options = ["A", "B+", "B", "C+", "C", "D+", "D", "F"]

with col_right:
    gr_info = st.selectbox("เกรดระบบสารสนเทศเบื้องต้น", grade_options, index=1)
    gr_arch = st.selectbox("เกรดโครงสร้างระบบคอมพิวเตอร์", grade_options, index=2)
    gr_prog = st.selectbox("เกรดการเขียนโปรแกรมคอมพิวเตอร์เบื้องต้น", grade_options, index=0)
    gr_mkt = st.selectbox("เกรดหลักการตลาด", grade_options, index=2)
    gr_logi = st.selectbox("เกรดโลจิสติกส์และการผลิต", grade_options, index=2)
    gr_biz = st.selectbox("เกรดโปรแกรมประยุกต์เพื่อทางธุรกิจ", grade_options, index=1)
    gr_net = st.selectbox("เกรดเทคโนโลยีอินเทอร์เน็ต", grade_options, index=1)
    gr_comm = st.selectbox("เกรดระบบการสื่อสารและเครือข่าย 1", grade_options, index=2)

st.markdown("---")

btn = st.button("🧠 ทำนายอาชีพที่เหมาะสม", use_container_width=True)

if btn:
    # Organize input data into a single row to match the model's feature order
    person = {
        "เพศ": gender,
        "ชั้นปี": int(year),
        "GPA": float(gpa),
        "ระบบสารสนเทศเบื้องต้น": gr_info,
        "โครงสร้างระบบคอมพิวเตอร์": gr_arch,
        "การเขียนโปรแกรมคอมพิวเตอร์เบื้องต้น": gr_prog,
        "หลักการตลาด": gr_mkt,
        "โลจิสติกส์และการผลิต": gr_logi,
        "โปรแกรมประยุกต์เพื่อทางธุรกิจ": gr_biz,
        "เทคโนโลยีอินเทอร์เน็ต": gr_net,
        "ระบบการสื่อสารและเครือข่าย 1": gr_comm,
    }
    
    # Create a DataFrame to prepare the input data for the model
    input_df = pd.DataFrame([person])

    # Convert gender to a one-hot encoded column to match the training data
    input_df['เพศ_หญิง'] = input_df['เพศ'].apply(lambda x: 1 if x == 'หญิง' else 0)
    input_df.drop('เพศ', axis=1, inplace=True)

    # Convert letter grades to numerical points
    for c in subject_cols:
        input_df[c] = input_df[c].map(grade_to_points)

    # Reorder columns to match the model's feature order
    final_input = input_df[feature_cols]

    # Make the prediction
    pred = model.predict(final_input)[0]
    proba = model.predict_proba(final_input)[0]
    proba_series = pd.Series(proba, index=model.classes_).sort_values(ascending=False)

    st.success(f"🎯 อาชีพที่เหมาะสม: **{pred}**")
    st.subheader("ความน่าจะเป็นแต่ละอาชีพ")
    st.bar_chart(proba_series)

    with st.expander("ดูข้อมูลอินพุต/ฟีเจอร์ที่ป้อน"):
        st.dataframe(final_input, use_container_width=True)