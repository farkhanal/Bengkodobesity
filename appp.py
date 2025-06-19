import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(
    page_title="Prediksi Tingkat Obesitas",
    page_icon=":bar_chart:",
    layout="centered",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #ffffff;
        color: #1f2937;
        font-family: 'Inter', sans-serif;
        max-width: 700px;
        margin: 2rem auto;
        padding: 2rem 3rem;
        border-radius: 12px;
        box-shadow: rgba(0,0,0,0.08) 0px 8px 24px;
    }
    h1 {
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #111827;
    }
    label {
        font-weight: 600;
        font-size: 16px;
        color: #374151;
    }
    .stButton>button {
        background-color: #111827;
        color: white;
        padding: 12px 28px;
        font-size: 18px;
        font-weight: 700;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main">', unsafe_allow_html=True)

# Tambahan nama kamu
st.markdown("""
<h2 style='text-align: center; margin-bottom: 0;'>Farkhan Al Fanani Ruwanto Putro</h2>
<hr style='margin-top: 0; margin-bottom: 2rem;'>
""", unsafe_allow_html=True)

st.title("Prediksi Tingkat Obesitas")
st.write(
    "Masukkan data diri dan kebiasaan Anda dengan lengkap untuk memprediksi tingkat obesitas berdasarkan model terpercaya."
)


st.subheader("Informasi Pribadi dan Kebiasaan")

def pilihan(label, options, penjelasan):
    st.write(f"**{label}**")
    for i, p in enumerate(penjelasan):
        st.caption(f"- {options[i]}: {p}")
    return st.selectbox(f"Pilih {label.lower()} Anda:", options)

gender_opts = ["Perempuan", "Laki-laki"]
gender_desc = ["Jenis kelamin wanita", "Jenis kelamin pria"]

calc_opts = ["Tidak pernah", "Kadang-kadang", "Sering", "Selalu"]
calc_desc = [
    "Tidak pernah mengonsumsi makanan tinggi kalori",
    "Mengonsumsi makanan tinggi kalori kadang-kadang",
    "Sering mengonsumsi makanan tinggi kalori",
    "Selalu mengonsumsi makanan tinggi kalori"
]

favc_opts = ["Tidak", "Ya"]
favc_desc = [
    "Tidak mengonsumsi makanan tinggi kalori secara rutin",
    "Mengonsumsi makanan tinggi kalori secara rutin"
]

scc_opts = ["Tidak", "Ya"]
scc_desc = [
    "Tidak mengonsumsi minuman bersoda atau manis",
    "Mengonsumsi minuman bersoda atau manis"
]

smoke_opts = ["Tidak", "Ya"]
smoke_desc = [
    "Tidak merokok",
    "Merokok"
]

family_history_opts = ["Tidak", "Ya"]
family_history_desc = [
    "Tidak memiliki riwayat keluarga kelebihan berat badan",
    "Memiliki riwayat keluarga kelebihan berat badan"
]

caec_opts = ["Tidak pernah", "Kadang-kadang", "Sering", "Selalu"]
caec_desc = [
    "Tidak pernah mengonsumsi alkohol",
    "Mengonsumsi alkohol kadang-kadang",
    "Sering mengonsumsi alkohol",
    "Selalu mengonsumsi alkohol"
]

mtrans_opts = [
    "Mobil pribadi",
    "Motor",
    "Transportasi umum",
    "Berjalan kaki",
    "Sepeda"
]
mtrans_desc = [
    "Menggunakan mobil pribadi",
    "Menggunakan motor",
    "Menggunakan transportasi umum",
    "Berjalan kaki sebagai moda transportasi",
    "Menggunakan sepeda"
]

age = st.number_input("Umur (tahun)", min_value=10, max_value=80, value=30, step=1)
height = st.number_input("Tinggi badan (cm)", min_value=140, max_value=210, value=170, step=1)
weight = st.number_input("Berat badan (kg)", min_value=30, max_value=200, value=70, step=1)
ncp = st.number_input("Jumlah makanan utama per hari (NCP)", min_value=1, max_value=6, value=3)
ch2o = st.number_input("Konsumsi air per hari (liter) (CH2O)", min_value=1.0, max_value=5.0, value=2.0, step=0.1, format="%.1f")
faf = st.number_input("Frekuensi aktivitas fisik per minggu (FAF)", min_value=0, max_value=20, value=3)
tue = st.number_input("Waktu menggunakan gadget per hari (jam) (TUE)", min_value=0, max_value=16, value=4)
fcvc = st.number_input("Frekuensi konsumsi sayur per minggu (FCVC)", min_value=0, max_value=21, value=3)

gender = pilihan("Jenis Kelamin (Gender)", gender_opts, gender_desc)
calc = pilihan("Konsumsi makanan tinggi kalori (CALC)", calc_opts, calc_desc)
favc = pilihan("Mengonsumsi makanan tinggi kalori secara rutin (FAVC)", favc_opts, favc_desc)
scc = pilihan("Konsumsi minuman bersoda atau manis (SCC)", scc_opts, scc_desc)
smoke = pilihan("Merokok (SMOKE)", smoke_opts, smoke_desc)
family_history = pilihan("Riwayat keluarga kelebihan berat badan (family_history_with_overweight)", family_history_opts, family_history_desc)
caec = pilihan("Konsumsi alkohol (CAEC)", caec_opts, caec_desc)
mtrans = pilihan("Moda transportasi utama (MTRANS)", mtrans_opts, mtrans_desc)

mapping_gender = {"Perempuan": 0, "Laki-laki": 1}
mapping_calc = {"Tidak pernah": 0, "Kadang-kadang": 1, "Sering": 2, "Selalu": 3}
mapping_favc = {"Tidak": 0, "Ya": 1}
mapping_scc = {"Tidak": 0, "Ya": 1}
mapping_smoke = {"Tidak": 0, "Ya": 1}
mapping_family_history = {"Tidak": 0, "Ya": 1}
mapping_caec = {"Tidak pernah": 0, "Kadang-kadang":1, "Sering": 2, "Selalu": 3}
mapping_mtrans = {
    "Mobil pribadi": 0,
    "Motor": 1,
    "Transportasi umum": 2,
    "Berjalan kaki": 3,
    "Sepeda": 4
}

input_data = {
    "Age": age,
    "Height": height,
    "Weight": weight,
    "NCP": ncp,
    "CH2O": ch2o,
    "FAF": faf,
    "TUE": tue,
    "FCVC": fcvc,
    "Gender": mapping_gender[gender],
    "CALC": mapping_calc[calc],
    "FAVC": mapping_favc[favc],
    "SCC": mapping_scc[scc],
    "SMOKE": mapping_smoke[smoke],
    "family_history_with_overweight": mapping_family_history[family_history],
    "CAEC": mapping_caec[caec],
    "MTRANS": mapping_mtrans[mtrans]
}

input_df = pd.DataFrame([input_data], columns=[
    "Age", "Height", "Weight", "NCP", "CH2O", "FAF", "TUE", "FCVC",
    "Gender", "CALC", "FAVC", "SCC", "SMOKE", "family_history_with_overweight",
    "CAEC", "MTRANS"
])

@st.cache_data(show_spinner=False)
def load_and_train_model():
    df = pd.read_csv('ObesityDataSet.csv')

    cols_with_question_mark = [
        'Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE',
        'family_history_with_overweight', 'CAEC', 'MTRANS'
    ]
    df = df[~df[cols_with_question_mark].isin(['?']).any(axis=1)].copy()

    numeric_cols = ['Age', 'Height', 'Weight', 'NCP', 'CH2O', 'FAF', 'TUE', 'FCVC']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    target_col = "NObeyesdad"
    cat_cols = [col for col in df.select_dtypes(include=['object']).columns if col != target_col]
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    target_le = LabelEncoder()
    df[target_col] = target_le.fit_transform(df[target_col])

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Simpan nama fitur training untuk validasi nanti 
    feature_names = list(X.columns)

    return model, target_le, scaler, numeric_cols, feature_names

model, target_le, scaler, numeric_cols, feature_names = load_and_train_model()

# Validasi dan susun ulang kolom input sesuai nama fitur training (untuk hindari error urutan/nama)
input_df = input_df.reindex(columns=feature_names)

input_df_scaled = input_df.copy()
input_df_scaled[numeric_cols] = scaler.transform(input_df[numeric_cols])

if st.button("Prediksi Tingkat Obesitas"):
    pred_encoded = model.predict(input_df_scaled)[0]
    pred_proba = model.predict_proba(input_df_scaled)[0]
    pred_label = target_le.inverse_transform([pred_encoded])[0]

    st.markdown("### Hasil Prediksi:")
    st.success(f"Tingkat obesitas Anda diprediksi adalah: **{pred_label}**")

    proba_df = pd.DataFrame({
        "Tingkat Obesitas": target_le.inverse_transform(np.arange(len(pred_proba))),
        "Probabilitas": pred_proba
    }).sort_values("Probabilitas", ascending=False)

    st.markdown("### Probabilitas Prediksi tiap Kelas:")
    st.dataframe(proba_df.style.format({"Probabilitas": "{:.2%}"}))

st.markdown("</div>", unsafe_allow_html=True)

