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
subject_cols = [c for c in feature_cols if c not in ["เพศ", "ชั้นปี", "GPA"]]

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
    gr_mkt  = st.selectbox("เกรดหลักการตลาด", grade_options, index=2)
    gr_logi = st.selectbox("เกรดโลจิสติกส์และการผลิต", grade_options, index=2)
    gr_biz  = st.selectbox("เกรดโปรแกรมประยุกต์เพื่อทางธุรกิจ", grade_options, index=1)
    gr_net  = st.selectbox("เกรดเทคโนโลยีอินเทอร์เน็ต", grade_options, index=1)
    gr_comm = st.selectbox("เกรดระบบการสื่อสารและเครือข่าย 1", grade_options, index=2)

st.markdown("---")

btn = st.button("🧠 ทำนายอาชีพที่เหมาะสม", use_container_width=True)

if btn:
    # จัดข้อมูลอินพุตเป็น 1 แถว ตามลำดับฟีเจอร์ของโมเดล
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
    row = pd.DataFrame([person])

    # แปลงเป็นตัวเลขให้ตรงกับฟีเจอร์ของโมเดล
    row["เพศ"] = row["เพศ"].map({"ชาย":0, "หญิง":1})
    for c in subject_cols:
        row[c] = row[c].map(grade_to_points)

    # จัดคอลัมน์ตาม model.feature_cols (กันหลงลำดับ)
    missing = [c for c in feature_cols if c not in row.columns]
    if missing:
        st.error(f"คอลัมน์หายไป: {missing}")
        st.stop()
    row = row[feature_cols]

    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    proba_series = pd.Series(proba, index=model.classes_).sort_values(ascending=False)

    st.success(f"🎯 อาชีพที่เหมาะสม: **{pred}**")
    st.subheader("ความน่าจะเป็นแต่ละอาชีพ")
    st.bar_chart(proba_series)

    with st.expander("ดูข้อมูลอินพุต/ฟีเจอร์ที่ป้อน"):
        st.dataframe(row, use_container_width=True)

st.sidebar.markdown("### ℹ️ เกี่ยวกับโมเดล")
st.sidebar.write(
    "- โมเดล: RandomForest (multi-class)\n"
    "- ฟีเจอร์: เพศ, ชั้นปี, GPA, เกรด 8 วิชา\n"
    "- เกรดจะแปลงเป็นคะแนน: A=4, B+=3.5, B=3, C+=2.5, C=2, D+=1.5, D=1, F=0"
)


