import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import plotly.io as pio

_tpl = go.layout.Template(pio.templates["plotly_white"])
_tpl.layout.update(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#1B3A24"),
    colorway=["#2E7D32", "#8DA14B", "#C9A227", "#4F7942", "#A3B18A", "#6B8E23"],
)
pio.templates["fiskal"] = _tpl
pio.templates.default = "fiskal"

st.set_page_config(page_title="Dashboard Fiskal 97 Kota", layout="wide")

TARGETS = ["KEMISKINAN", "PENGANGGURAN", "KETIMPANGAN", "IPM", "PE"]
LABEL = {
    "KEMISKINAN": "Kemiskinan (%)", "PENGANGGURAN": "Pengangguran (%)",
    "KETIMPANGAN": "Ketimpangan (Gini)", "IPM": "IPM", "PE": "Pertumbuhan Ekonomi (%)",
}

# ---------------------------------------------------------------- DATA
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

st.sidebar.title("Dashboard Fiskal")
upload = st.sidebar.file_uploader("Upload CSV (opsional)", type="csv")
try:
    df = load_data(upload if upload else "dataset_fiskal_97_hipoFull.csv")
except FileNotFoundError:
    st.error("File dataset_fiskal_97_hipoFull.csv tidak ditemukan. Upload lewat sidebar.")
    st.stop()

fitur_all = [c for c in df.columns if c not in ["Kota", "Tahun"] + TARGETS]

st.sidebar.header("Filter")
tahun_min, tahun_max = int(df.Tahun.min()), int(df.Tahun.max())
rentang = st.sidebar.slider("Rentang tahun", tahun_min, tahun_max, (tahun_min, tahun_max))
kota_pilih = st.sidebar.multiselect("Kota (kosong = semua)", sorted(df.Kota.unique()))

dff = df[df.Tahun.between(*rentang)]
if kota_pilih:
    dff = dff[dff.Kota.isin(kota_pilih)]
if dff.empty:
    st.warning("Tidak ada data untuk filter ini.")
    st.stop()

st.title("Dashboard Data Panel Fiskal & Pembangunan Kota")
st.caption(f"{dff.Kota.nunique()} kota • {dff.Tahun.min()}–{dff.Tahun.max()} • {len(dff):,} observasi")

tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan", "Eksplorasi", "Machine Learning", "Data"])

# ---------------------------------------------------------------- TAB 1
with tab1:
    k = st.columns(5)
    for col, t in zip(k, TARGETS):
        awal = dff[dff.Tahun == dff.Tahun.min()][t].mean()
        akhir = dff[dff.Tahun == dff.Tahun.max()][t].mean()
        col.metric(LABEL[t], f"{dff[t].mean():.2f}", f"{akhir - awal:+.2f} (awal→akhir)")

    c1, c2 = st.columns(2)
    ind = c1.selectbox("Indikator tren", TARGETS, format_func=LABEL.get)
    tren = dff.groupby("Tahun")[ind].mean().reset_index()
    c1.plotly_chart(px.line(tren, x="Tahun", y=ind, markers=True,
                            title=f"Rata-rata {LABEL[ind]} per tahun"), use_container_width=True, theme=None)

    tahun_rank = c2.selectbox("Tahun peringkat", sorted(dff.Tahun.unique(), reverse=True))
    rank = dff[dff.Tahun == tahun_rank].nlargest(10, ind)[["Kota", ind]]
    c2.plotly_chart(px.bar(rank.sort_values(ind), x=ind, y="Kota", orientation="h",
                           title=f"10 kota tertinggi – {LABEL[ind]} ({tahun_rank})"),
                    use_container_width=True, theme=None)

    st.info(
        f"**Interpretasi:** rata-rata {LABEL[ind]} pada {tren.Tahun.iloc[0]} adalah "
        f"{tren[ind].iloc[0]:.2f} dan pada {tren.Tahun.iloc[-1]} adalah {tren[ind].iloc[-1]:.2f} "
        f"(selisih {tren[ind].iloc[-1] - tren[ind].iloc[0]:+.2f}). "
        "Perhatikan apakah perubahan ini konsisten tiap tahun atau hanya fluktuasi kecil."
    )

# ---------------------------------------------------------------- TAB 2
with tab2:
    st.subheader("Korelasi")
    kolom_kor = st.multiselect("Variabel", fitur_all + TARGETS, default=["PAD", "DAU_Grant", "ICT"] + TARGETS)
    if len(kolom_kor) >= 2:
        cm = dff[kolom_kor].corr()
        st.plotly_chart(px.imshow(cm, text_auto=".2f", color_continuous_scale="BrBG",
                                  zmin=-1, zmax=1, aspect="auto"), use_container_width=True, theme=None)
        pairs = (cm.where(np.triu(np.ones(cm.shape), 1).astype(bool)).stack()
                   .sort_values(key=abs, ascending=False))
        if len(pairs):
            (a, b), r = pairs.index[0], pairs.iloc[0]
            kuat = "kuat" if abs(r) >= .6 else "sedang" if abs(r) >= .3 else "lemah/hampir tidak ada"
            st.info(f"**Interpretasi:** pasangan terkuat adalah **{a}–{b}** (r = {r:.2f}), "
                    f"tergolong hubungan {kuat}. Korelasi ≠ kausalitas.")

    st.subheader("Scatter plot")
    s1, s2, s3 = st.columns(3)
    x = s1.selectbox("Sumbu X", fitur_all)
    y = s2.selectbox("Sumbu Y", TARGETS, format_func=LABEL.get)
    warna = s3.checkbox("Warnai per tahun")
    st.plotly_chart(px.scatter(dff, x=x, y=y, color="Tahun" if warna else None,
                               hover_name="Kota", trendline="ols", opacity=.6),
                    use_container_width=True, theme=None)
    r = dff[[x, y]].corr().iloc[0, 1]
    st.info(f"**Interpretasi:** r({x}, {y}) = {r:.3f}. "
            + ("Ada pola linear yang cukup jelas." if abs(r) >= .3
               else "Titik menyebar acak; hubungan linear antara keduanya sangat lemah."))

# ---------------------------------------------------------------- TAB 3
with tab3:
    st.subheader("Model prediksi")
    m1, m2, m3 = st.columns(3)
    target = m1.selectbox("Target (Y)", TARGETS, format_func=LABEL.get, key="tg")
    algo = m2.selectbox("Algoritma", ["Random Forest", "Regresi Linear"])
    split = m3.selectbox("Cara split", ["Acak 80/20", "Berdasarkan waktu (uji = 2 tahun terakhir)"])
    fitur = st.multiselect("Fitur (X)", fitur_all, default=fitur_all)

    if len(fitur) < 1:
        st.warning("Pilih minimal satu fitur.")
        st.stop()

    # model dilatih dari seluruh df (bukan dff) supaya stabil terhadap filter
    if split.startswith("Acak"):
        Xtr, Xte, ytr, yte = train_test_split(df[fitur], df[target], test_size=.2, random_state=42)
    else:
        batas = df.Tahun.max() - 2
        tr, te = df[df.Tahun <= batas], df[df.Tahun > batas]
        Xtr, ytr, Xte, yte = tr[fitur], tr[target], te[fitur], te[target]

    @st.cache_resource
    def latih(algo, fitur, target, split):
        model = (RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1)
                 if algo == "Random Forest" else LinearRegression())
        return model.fit(Xtr, ytr)

    model = latih(algo, tuple(fitur), target, split)
    pred = model.predict(Xte)
    r2 = r2_score(yte, pred)
    mae = mean_absolute_error(yte, pred)
    rmse = mean_squared_error(yte, pred) ** .5
    base_mae = mean_absolute_error(yte, np.full(len(yte), ytr.mean()))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("R² (data uji)", f"{r2:.3f}")
    k2.metric("MAE", f"{mae:.3f}")
    k3.metric("RMSE", f"{rmse:.3f}")
    k4.metric("MAE baseline (tebak rata-rata)", f"{base_mae:.3f}")

    g1, g2 = st.columns(2)
    fig = px.scatter(x=yte, y=pred, labels={"x": "Aktual", "y": "Prediksi"}, opacity=.6,
                     title="Aktual vs Prediksi")
    lo, hi = float(min(yte.min(), pred.min())), float(max(yte.max(), pred.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="ideal", line=dict(dash="dash")))
    g1.plotly_chart(fig, use_container_width=True, theme=None)

    imp = permutation_importance(model, Xte, yte, n_repeats=10, random_state=42, n_jobs=-1)
    imp_df = (pd.DataFrame({"Fitur": fitur, "Pentingnya": imp.importances_mean, "SD": imp.importances_std})
              .sort_values("Pentingnya"))
    g2.plotly_chart(px.bar(imp_df.tail(10), x="Pentingnya", y="Fitur", orientation="h",
                           error_x="SD", title="Permutation importance (10 teratas)"),
                    use_container_width=True, theme=None)

    # ---- interpretasi otomatis
    if r2 >= .6:
        kualitas = "**baik**: model menjelaskan sebagian besar variasi data uji."
    elif r2 >= .3:
        kualitas = "**sedang**: ada pola, tetapi masih banyak variasi yang tidak terjelaskan."
    elif r2 >= .1:
        kualitas = "**lemah**: daya prediksi rendah."
    else:
        kualitas = ("**sangat lemah / tidak ada**: model tidak lebih baik daripada menebak rata-rata. "
                    "Artinya fitur yang dipakai tidak cukup menjelaskan target pada data ini.")
    teratas = imp_df.iloc[-1]
    st.info(
        f"**Interpretasi:** R² = {r2:.3f} → kemampuan prediksi {kualitas}  \n"
        f"Rata-rata kesalahan (MAE) {mae:.3f} vs {base_mae:.3f} bila hanya menebak rata-rata "
        f"({'lebih baik' if mae < base_mae else 'tidak lebih baik'}).  \n"
        f"Fitur dengan kontribusi terbesar: **{teratas.Fitur}** "
        f"(penurunan R² bila diacak = {teratas.Pentingnya:.4f}). "
        + ("Nilai importance mendekati 0 untuk semua fitur berarti tidak ada fitur yang benar-benar berpengaruh."
           if imp_df.Pentingnya.max() < .01 else "")
    )
    if algo == "Regresi Linear":
        koef = pd.DataFrame({"Fitur": fitur, "Koefisien": model.coef_}).sort_values("Koefisien")
        st.dataframe(koef, use_container_width=True)
        st.caption("Koefisien = perubahan prediksi target bila fitur naik 1 satuan (fitur lain tetap).")

    # ---- simulasi prediksi (what-if)
    st.subheader("Simulasi prediksi")
    cols = st.columns(4)
    input_user = {}
    for i, f in enumerate(fitur):
        input_user[f] = cols[i % 4].slider(f, float(df[f].min()), float(df[f].max()), float(df[f].median()))
    hasil = model.predict(pd.DataFrame([input_user]))[0]
    st.success(f"Prediksi {LABEL[target]}: **{hasil:.2f}**  (±{rmse:.2f} perkiraan galat)")

# ---------------------------------------------------------------- TAB 4
with tab4:
    st.dataframe(dff, use_container_width=True)
    st.download_button("Unduh data terfilter (CSV)", dff.to_csv(index=False).encode(), "data_terfilter.csv")