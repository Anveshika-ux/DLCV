from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageFilter, ImageOps

EPS = 1e-8
METHODS = ["LBP", "NN", "DNN", "CNN", "Color Features", "Proposed", "Hybrid"]
HANDCRAFT_METHODS = {"LBP", "Color Features"}
OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
SOURCE_NOTEBOOKS = {
    "LBP": r"C:\Users\HP\Desktop\DLCV\Task 1",
    "NN": r"C:\Users\HP\Desktop\DLCV\Task 2",
    "DNN": r"C:\Users\HP\Desktop\DLCV\Task 2",
    "CNN": r"C:\Users\HP\Desktop\DLCV\Task 3",
    "Proposed": r"C:\Users\HP\Desktop\DLCV\Task 4",
    "Hybrid": r"C:\Users\HP\Desktop\DLCV\task 5",
    "Color Features": r"C:\Users\HP\Desktop\DLCV\Task 6",
}
METHOD_REMARKS = {
    "LBP": "Task 1 texture histogram baseline",
    "NN": "Task 2 shallow dense embedding",
    "DNN": "Task 2 deeper dense embedding",
    "CNN": "Task 3 convolutional feature learning",
    "Color Features": "Task 6 intra/inter color statistics",
    "Proposed": "Task 4 edge + FFT + texture fusion",
    "Hybrid": "Task 5 weighted CNN + proposed fusion",
}
NOTEBOOK_FILES = {
    "LBP": ["LBP_CALTECH.ipynb", "LBP_CIFAR.ipynb", "LBP_MNIST.ipynb"],
    "NN": ["NN&DNN_CALTECH (1).ipynb", "NN&DNN_CIFAR.ipynb", "NN&DNN_MNIST.ipynb"],
    "DNN": ["NN&DNN_CALTECH (1).ipynb", "NN&DNN_CIFAR.ipynb", "NN&DNN_MNIST.ipynb"],
    "CNN": ["CNN_CALTECH.ipynb", "CNN_CIFAR.ipynb", "CNN_MNIST.ipynb"],
    "Proposed": ["Novel_Caltech.ipynb", "Novel_Cifar.ipynb", "Novel_MNIST.ipynb"],
    "Hybrid": ["HYBRID_CALTECH.ipynb", "HYBRID_CIFAR.ipynb"],
    "Color Features": ["Feature.ipynb"],
}
DATASETS = {
    "CIFAR-10": {
        "size": 32,
        "ch": 3,
        "gallery": 700,
        "queries": 120,
        "train": 1200,
        "classes": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    },
    "MNIST": {
        "size": 32,
        "ch": 1,
        "gallery": 900,
        "queries": 180,
        "train": 1600,
        "classes": [str(i) for i in range(10)],
    },
    "Fashion-MNIST": {
        "size": 32,
        "ch": 1,
        "gallery": 900,
        "queries": 180,
        "train": 1600,
        "classes": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
    },
}


def setup():
    st.set_page_config("Task 7 Dashboard", "IR", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(
        """
        <style>
        .stApp {background:#f7f8fc;color:#17212b;}
        .block-container {max-width:1200px;padding-top:1.5rem;padding-bottom:2rem;}
        h1,h2,h3,h4,h5,h6,p,span,label,div {color:#17212b !important;}
        .hero,.card,.mcard{background:#fff;border:1px solid #dde5f0;border-radius:18px;box-shadow:0 8px 24px rgba(15,23,42,.05);}
        .hero{padding:1.35rem 1.5rem;margin-bottom:1rem;}
        .card{padding:1rem;margin-bottom:1rem;}
        .mcard{padding:1rem 1.1rem;text-align:center;min-height:118px;}
        .t1{font-size:2rem;font-weight:700;margin:0 0 .3rem 0;}
        .t2{font-size:1rem;color:#475569 !important;margin:0;}
        .ml{font-size:.85rem;font-weight:600;color:#475569 !important;}
        .mv{font-size:1.34rem;font-weight:700;color:#0f172a !important;white-space:nowrap;line-height:1.15;}
        .ms{font-size:.78rem;color:#64748b !important;}
        div[data-testid="stWidgetLabel"] p {color:#0f172a !important;font-weight:600 !important;}
        div[data-baseweb="select"] > div {background:#ffffff !important;color:#0f172a !important;border-color:#cbd5e1 !important;}
        div[data-baseweb="select"] * {color:#0f172a !important;}
        ul[role="listbox"] *, div[role="listbox"] * {color:#0f172a !important;background:#ffffff !important;}
        div[data-testid="stFileUploader"] section {background:#ffffff !important;border:1px dashed #94a3b8 !important;}
        div[data-testid="stFileUploader"] section * {color:#0f172a !important;}
        .stTextInput input, .stNumberInput input, textarea {background:#ffffff !important;color:#0f172a !important;}
        div[data-baseweb="slider"] * {color:#0f172a !important;}
        .stButton > button {background:#0f766e !important;color:#ffffff !important;border:1px solid #0f766e !important;border-radius:12px !important;}
        .stButton > button:hover {background:#115e59 !important;border-color:#115e59 !important;}
        .stRadio label, .stRadio p {color:#0f172a !important;}
        .stInfo, .stAlert {color:#0f172a !important;}
        div[data-baseweb="popover"] {background:#ffffff !important;color:#0f172a !important;}
        div[data-baseweb="popover"] * {color:#0f172a !important;}
        div[data-baseweb="popover"] ul {background:#ffffff !important;border:1px solid #cbd5e1 !important;}
        div[data-baseweb="popover"] li {background:#ffffff !important;color:#0f172a !important;}
        div[data-baseweb="popover"] li:hover {background:#e8f0fe !important;color:#0f172a !important;}
        div[role="option"] {background:#ffffff !important;color:#0f172a !important;}
        div[role="option"] * {color:#0f172a !important;}
        div[role="option"][aria-selected="true"] {background:#dbeafe !important;color:#0f172a !important;}
        </style>
        <div class="hero">
            <div class="t1">Task 7: Image Retrieval Dashboard</div>
            <p class="t2">Choose dataset, query source, Top-K, and retrieval model. Then retrieve similar images and generate the comparative analysis table required in the assignment.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def tfmod():
    import tensorflow as tf

    return tf


def nvec(v):
    return v.astype(np.float32) / max(float(np.linalg.norm(v)), EPS)


def nrows(m):
    return m.astype(np.float32) / np.clip(np.linalg.norm(m, axis=1, keepdims=True), EPS, None)


def prep(pil, ds):
    cfg = DATASETS[ds]
    size, ch = cfg["size"], cfg["ch"]
    img = ImageOps.exif_transpose(pil)
    if ch == 1:
        disp = ImageOps.pad(img.convert("L").convert("RGB"), (size, size), method=Image.Resampling.BILINEAR, color=(0, 0, 0))
        model = ImageOps.pad(img.convert("L"), (size, size), method=Image.Resampling.BILINEAR, color=0)
        return np.asarray(disp, dtype=np.uint8), (np.asarray(model, dtype=np.float32)[..., None] / 255.0).astype(np.float32)
    disp = ImageOps.pad(img.convert("RGB"), (size, size), method=Image.Resampling.BILINEAR, color=(0, 0, 0))
    return np.asarray(disp, dtype=np.uint8), (np.asarray(disp, dtype=np.float32) / 255.0).astype(np.float32)


def prep_batch(raw, ds):
    ch = DATASETS[ds]["ch"]
    d, m = [], []
    for arr in raw:
        pil = Image.fromarray(arr.astype(np.uint8)).convert("L" if ch == 1 else "RGB")
        di, mo = prep(pil, ds)
        d.append(di)
        m.append(mo)
    return np.stack(d).astype(np.uint8), np.stack(m).astype(np.float32)


@st.cache_data(show_spinner=False)
def load_bundle(ds, gallery, queries, train_n):
    tf = tfmod()
    if ds == "CIFAR-10":
        (xtr, ytr), (xte, yte) = tf.keras.datasets.cifar10.load_data()
        ytr, yte = ytr.flatten(), yte.flatten()
    elif ds == "MNIST":
        (xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
    else:
        (xtr, ytr), (xte, yte) = tf.keras.datasets.fashion_mnist.load_data()
    rng = np.random.default_rng(42)
    ti = rng.choice(len(xtr), size=min(train_n, len(xtr)), replace=False)
    gi = rng.choice(len(xtr), size=min(gallery, len(xtr)), replace=False)
    qi = rng.choice(len(xte), size=min(queries, len(xte)), replace=False)
    trd, trm = prep_batch(xtr[ti], ds)
    gd, gm = prep_batch(xtr[gi], ds)
    qd, qm = prep_batch(xte[qi], ds)
    return {"train_d": trd, "train_m": trm, "train_y": np.asarray(ytr[ti], np.int32), "gal_d": gd, "gal_m": gm, "gal_y": np.asarray(ytr[gi], np.int32), "qry_d": qd, "qry_m": qm, "qry_y": np.asarray(yte[qi], np.int32), "classes": DATASETS[ds]["classes"]}


def gray(rgb):
    x = rgb.astype(np.float32) / 255.0
    return (0.2989 * x[:, :, 0] + 0.5870 * x[:, :, 1] + 0.1140 * x[:, :, 2]).astype(np.float32)


def corr(a, b):
    a, b = a.ravel().astype(np.float32), b.ravel().astype(np.float32)
    a, b = a - a.mean(), b - b.mean()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if d < EPS else float(np.dot(a, b) / d)


def skewness(x):
    x = x.astype(np.float32).ravel()
    c = x - x.mean()
    s = float(x.std())
    return 0.0 if s < EPS else float(np.mean((c / s) ** 3))


def kurtosis_excess(x):
    x = x.astype(np.float32).ravel()
    c = x - x.mean()
    s = float(x.std())
    return 0.0 if s < EPS else float(np.mean((c / s) ** 4) - 3.0)


def mi(a, b, bins=16):
    a = np.clip(a * 255.0, 0, 255).astype(np.uint8).ravel()
    b = np.clip(b * 255.0, 0, 255).astype(np.uint8).ravel()
    h, _, _ = np.histogram2d(a, b, bins=bins, range=[[0, 255], [0, 255]])
    h = h.astype(np.float64)
    h /= h.sum() + EPS
    px, py = h.sum(axis=1, keepdims=True), h.sum(axis=0, keepdims=True)
    return float(np.sum(h * np.log(np.clip(h / np.clip(px @ py, EPS, None), EPS, None))))


def feat_lbp(rgb):
    g = gray(rgb)
    p = np.pad(g, 1, mode="edge")
    c = p[1:-1, 1:-1]
    out = np.zeros_like(c, dtype=np.uint8)
    for bit, (dy, dx) in enumerate(OFFSETS):
        n = p[1 + dy : 1 + dy + c.shape[0], 1 + dx : 1 + dx + c.shape[1]]
        out |= ((n >= c).astype(np.uint8) << bit)
    h = np.bincount(out.ravel(), minlength=256).astype(np.float32)
    return h / (h.sum() + EPS)


def feat_color(rgb):
    x = rgb.astype(np.float32) / 255.0
    intra = []
    for c in range(3):
        p = x[:, :, c].ravel()
        intra += [float(p.mean()), float(p.std()), skewness(p), kurtosis_excess(p)]
    r, g, b = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    inter = [corr(r, g), corr(g, b), corr(r, b), mi(r, g), mi(g, b), mi(r, b)]
    return nvec(np.asarray(intra + inter, np.float32))


def feat_one(rgb, method):
    if method == "LBP":
        return feat_lbp(rgb)
    if method == "Color Features":
        return feat_color(rgb)
    raise ValueError(method)


def feat_mat(imgs, method):
    return nrows(np.vstack([feat_one(img, method) for img in imgs]).astype(np.float32))


def distort(img, mode):
    pil = Image.fromarray(img.astype(np.uint8))
    if mode == "noise":
        n = np.random.default_rng(42).normal(0, 18, size=img.shape)
        return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    if mode == "blur":
        return np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=1.4)), dtype=np.uint8)
    return np.asarray(pil.rotate(12, resample=Image.Resampling.BILINEAR), dtype=np.uint8)


def eval_split(qf, qy, gf, gy, k):
    p, r, ap, nd = [], [], [], []
    for i in range(len(qy)):
        idx = np.argsort(-(gf @ nvec(qf[i])))[:k]
        rel = (gy[idx] == qy[i]).astype(np.int32)
        tot = int((gy == qy[i]).sum())
        p.append(float(rel.sum() / max(k, 1)))
        r.append(float(rel.sum() / max(tot, 1)))
        hits = 0
        s = 0.0
        dcg = 0.0
        for rank, x in enumerate(rel, 1):
            if x:
                hits += 1
                s += hits / rank
                dcg += 1.0 / np.log2(rank + 1)
        ap.append(float(s / max(tot, 1)))
        idcg = sum(1.0 / np.log2(j + 1) for j in range(1, min(tot, len(rel)) + 1))
        nd.append(float(dcg / max(idcg, EPS)))
    return {"Precision": np.mean(p), "Recall": np.mean(r), "mAP": np.mean(ap), "RQ": np.mean(nd)}


def mk_nn(tf, shape, n):
    i = tf.keras.Input(shape=shape)
    x = tf.keras.layers.Flatten()(i)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(96, activation="relu", name="emb")(x)
    o = tf.keras.layers.Dense(n, activation="softmax")(x)
    m = tf.keras.Model(i, o)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def mk_dnn(tf, shape, n):
    i = tf.keras.Input(shape=shape)
    x = tf.keras.layers.Flatten()(i)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="emb")(x)
    o = tf.keras.layers.Dense(n, activation="softmax")(x)
    m = tf.keras.Model(i, o)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def mk_cnn(tf, shape, n):
    i = tf.keras.Input(shape=shape)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(i)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="emb")(x)
    o = tf.keras.layers.Dense(n, activation="softmax")(x)
    m = tf.keras.Model(i, o)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def mk_proposed(tf, shape, n):
    class ProposedExtractor(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.edge_dense = tf.keras.layers.Dense(128)
            self.fft_pool = tf.keras.layers.GlobalAveragePooling2D()
            self.fft_dense = tf.keras.layers.Dense(128)
            self.texture_dense = tf.keras.layers.Dense(128)

        def call(self, inputs):
            edges = tf.image.sobel_edges(inputs)
            edges = tf.sqrt(tf.reduce_sum(tf.square(edges), axis=-1))
            edges = tf.reduce_mean(edges, axis=[1, 2, 3])
            edges = tf.expand_dims(edges, -1)
            f_e = self.edge_dense(edges)

            gray = inputs if inputs.shape[-1] == 1 else tf.image.rgb_to_grayscale(inputs)
            fft = tf.signal.fft2d(tf.cast(gray, tf.complex64))
            fft = tf.math.abs(fft)
            fft = self.fft_pool(fft)
            f_f = self.fft_dense(fft)

            mean = tf.reduce_mean(gray, axis=[1, 2])
            var = tf.math.reduce_variance(gray, axis=[1, 2])
            texture = tf.concat([mean, var], axis=-1)
            f_t = self.texture_dense(texture)

            feat = tf.concat([f_e, f_f, f_t], axis=-1)
            return tf.nn.l2_normalize(feat, axis=1)

    i = tf.keras.Input(shape=shape)
    x = ProposedExtractor(name="emb")(i)
    o = tf.keras.layers.Dense(n, activation="softmax")(x)
    m = tf.keras.Model(i, o)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


@st.cache_resource(show_spinner=False)
def learned_pack(ds, gallery, queries, train_n):
    tf = tfmod()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    b = load_bundle(ds, gallery, queries, train_n)
    shape = tuple(b["train_m"].shape[1:])
    n = len(b["classes"])
    qd = {m: np.stack([distort(x, m) for x in b["qry_d"]]).astype(np.uint8) for m in ["noise", "blur", "rotate"]}
    qdm = {m: (np.mean(qd[m].astype(np.float32), axis=-1, keepdims=True) / 255.0 if DATASETS[ds]["ch"] == 1 else qd[m].astype(np.float32) / 255.0).astype(np.float32) for m in qd}
    out = {}
    for name, builder in [("NN", mk_nn), ("DNN", mk_dnn), ("CNN", mk_cnn), ("Proposed", mk_proposed)]:
        model = builder(tf, shape, n)
        model.fit(b["train_m"], b["train_y"], epochs=2, batch_size=64, verbose=0)
        emb = tf.keras.Model(model.input, model.get_layer("emb").output)
        out[name] = {
            "emb": emb,
            "gallery": nrows(emb.predict(b["gal_m"], verbose=0)),
            "query": nrows(emb.predict(b["qry_m"], verbose=0)),
            "noise": nrows(emb.predict(qdm["noise"], verbose=0)),
            "blur": nrows(emb.predict(qdm["blur"], verbose=0)),
            "rotate": nrows(emb.predict(qdm["rotate"], verbose=0)),
        }
    h = {m: {"gallery": feat_mat(b["gal_d"], m), "query": feat_mat(b["qry_d"], m), "noise": feat_mat(qd["noise"], m), "blur": feat_mat(qd["blur"], m), "rotate": feat_mat(qd["rotate"], m)} for m in HANDCRAFT_METHODS}
    lam = 0.6
    d = min(out["CNN"]["gallery"].shape[1], out["Proposed"]["gallery"].shape[1])
    out["Hybrid"] = {
        "emb": None,
        "gallery": nrows(lam * out["CNN"]["gallery"][:, :d] + (1 - lam) * out["Proposed"]["gallery"][:, :d]),
        "query": nrows(lam * out["CNN"]["query"][:, :d] + (1 - lam) * out["Proposed"]["query"][:, :d]),
        "noise": nrows(lam * out["CNN"]["noise"][:, :d] + (1 - lam) * out["Proposed"]["noise"][:, :d]),
        "blur": nrows(lam * out["CNN"]["blur"][:, :d] + (1 - lam) * out["Proposed"]["blur"][:, :d]),
        "rotate": nrows(lam * out["CNN"]["rotate"][:, :d] + (1 - lam) * out["Proposed"]["rotate"][:, :d]),
    }
    out["hand"] = h
    return out


def qfeat(qd, qm, method, pack):
    if method in HANDCRAFT_METHODS:
        return nvec(feat_one(qd, method))
    if method == "Hybrid":
        p = nvec(pack["Proposed"]["emb"].predict(qm[None, ...], verbose=0)[0].astype(np.float32))
        c = nvec(pack["CNN"]["emb"].predict(qm[None, ...], verbose=0)[0].astype(np.float32))
        d = min(len(c), len(p))
        return nvec(0.6 * c[:d] + 0.4 * p[:d])
    return nvec(pack[method]["emb"].predict(qm[None, ...], verbose=0)[0].astype(np.float32))


@st.cache_data(show_spinner=False)
def assignment_table(ds, gallery, queries, train_n, k):
    b = load_bundle(ds, gallery, queries, train_n)
    p = learned_pack(ds, gallery, queries, train_n)
    rows = []
    for method in METHODS:
        f = p["hand"][method] if method in HANDCRAFT_METHODS else p[method]
        clean = eval_split(f["query"], b["qry_y"], f["gallery"], b["gal_y"], k)
        robust = np.mean([eval_split(f["noise"], b["qry_y"], f["gallery"], b["gal_y"], k)["mAP"], eval_split(f["blur"], b["qry_y"], f["gallery"], b["gal_y"], k)["mAP"], eval_split(f["rotate"], b["qry_y"], f["gallery"], b["gal_y"], k)["mAP"]])
        rows.append({"Method": method, "Precision": round(clean["Precision"], 4), "Recall": round(clean["Recall"], 4), "mAP": round(clean["mAP"], 4), "Robustness": round(float(robust), 4)})
    best = max(r["mAP"] for r in rows) if rows else 0.0
    for r in rows:
        r["Remarks"] = "Best overall retrieval" if abs(r["mAP"] - best) < 1e-6 else ("Strong under distortions" if r["Robustness"] >= 0.60 else METHOD_REMARKS[r["Method"]])
    return pd.DataFrame(rows)


def metric_card(title, value, sub=""):
    st.markdown(f'<div class="mcard"><div class="ml">{title}</div><div class="mv">{value}</div><div class="ms">{sub}</div></div>', unsafe_allow_html=True)


def gallery_features_for_method(method, pack):
    return pack["hand"][method]["gallery"] if method in HANDCRAFT_METHODS else pack[method]["gallery"]


def query_payload(req, bundle, pack):
    if req["query_source"] == "Dataset sample":
        sample_idx = int(req["sample_index"])
        qd = bundle["qry_d"][sample_idx]
        qm = bundle["qry_m"][sample_idx]
        label_idx = int(bundle["qry_y"][sample_idx])
        label_name = bundle["classes"][label_idx]
        if req["method"] in HANDCRAFT_METHODS:
            qf = pack["hand"][req["method"]]["query"][sample_idx]
        else:
            qf = pack[req["method"]]["query"][sample_idx]
        return {
            "query_display": qd,
            "query_model": qm,
            "query_feature": qf,
            "label_idx": label_idx,
            "label_name": label_name,
            "caption": f"Dataset Query Sample #{sample_idx}",
        }

    qd, qm = prep(Image.open(io.BytesIO(req["bytes"])), req["ds"])
    label_name = req["label"] if req["label"] is not None else "Unknown"
    label_idx = None if req["label"] is None else bundle["classes"].index(req["label"])
    return {
        "query_display": qd,
        "query_model": qm,
        "query_feature": qfeat(qd, qm, req["method"], pack),
        "label_idx": label_idx,
        "label_name": label_name,
        "caption": "Uploaded Query Image",
    }


def main():
    setup()
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0
    if "sample_key_version" not in st.session_state:
        st.session_state["sample_key_version"] = 0

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1. Configure Retrieval")
    c1, c2, c3 = st.columns([1, 1, .8])
    ds = c1.selectbox("Dataset", list(DATASETS.keys()), index=0)
    method = c2.selectbox("Method for Retrieved Images", METHODS, index=6)
    k = c3.slider("Top-K", 3, 10, 5)
    cfg = DATASETS[ds]
    c4, c5, c6 = st.columns(3)
    gallery = c4.slider("Gallery Size", 300, 1500, cfg["gallery"], 100)
    queries = c5.slider("Benchmark Queries", 60, 300, cfg["queries"], 20)
    train_n = c6.slider("Training Images for NN/DNN/CNN", 600, 2500, cfg["train"], 100)

    qsrc_cols = st.columns([1.2, 0.8])
    query_source = qsrc_cols[0].radio("Query Source", ["Dataset sample", "Upload external image"], horizontal=True)
    qsrc_cols[1].markdown(
        """
        <div style="padding-top:1.8rem;color:#475569;font-size:0.92rem;">
        Use <strong>Dataset sample</strong> for an in-dataset query with known label,
        or <strong>Upload external image</strong> for a custom query.
        </div>
        """,
        unsafe_allow_html=True,
    )

    chosen_label = None
    uploaded_bytes = None
    sample_index = 0

    if query_source == "Dataset sample":
        preview_bundle = load_bundle(ds, gallery, queries, train_n)
        max_index = len(preview_bundle["qry_d"]) - 1
        sample_widget_key = f"sample_index_{st.session_state['sample_key_version']}"
        if sample_widget_key not in st.session_state or st.session_state[sample_widget_key] > max_index:
            st.session_state[sample_widget_key] = 0
        s1, s2, s3 = st.columns([1.05, 0.35, 0.9])
        s1.slider("Choose Dataset Query Image", 0, max_index, key=sample_widget_key)
        sample_index = int(st.session_state[sample_widget_key])
        if s2.button("Random", use_container_width=True):
            next_version = st.session_state["sample_key_version"] + 1
            st.session_state["sample_key_version"] = next_version
            st.session_state[f"sample_index_{next_version}"] = int(np.random.default_rng().integers(0, max_index + 1))
            st.rerun()
        chosen_label = preview_bundle["classes"][int(preview_bundle["qry_y"][sample_index])]
        s3.image(
            preview_bundle["qry_d"][sample_index],
            width="content",
            caption=f"Selected dataset query | Label: {chosen_label}",
        )
    else:
        c7, c8 = st.columns([1.15, .85])
        uploaded_file = c7.file_uploader(
            "Upload Query Image",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            key=f"uploader_{st.session_state['uploader_key']}",
        )
        chosen = c8.selectbox("True Label (optional)", ["Unknown"] + cfg["classes"], 0)
        uploaded_bytes = None if uploaded_file is None else uploaded_file.getvalue()
        chosen_label = None if chosen == "Unknown" else chosen

    action_cols = st.columns([0.34, 0.24, 1.0])
    run = action_cols[0].button(
        "Generate Dashboard Results",
        type="primary",
        disabled=query_source == "Upload external image" and uploaded_bytes is None,
        use_container_width=True,
    )
    reset = action_cols[1].button("Reset Dashboard", use_container_width=True)
    if reset:
        st.session_state.pop("req", None)
        st.session_state["uploader_key"] += 1
        next_version = st.session_state["sample_key_version"] + 1
        st.session_state["sample_key_version"] = next_version
        st.session_state[f"sample_index_{next_version}"] = 0
        st.rerun()

    st.markdown(
        '<p style="color:#475569;margin-top:.5rem;">The table at the end is generated for all methods on the selected dataset using the current Top-K setting.</p>',
        unsafe_allow_html=True,
    )
    with st.expander("Method-to-task mapping used in this dashboard"):
        src_df = pd.DataFrame(
            [
                {
                    "Method": method_name,
                    "Task Folder": SOURCE_NOTEBOOKS[method_name].split("\\")[-1],
                    "Notebook Files": ", ".join(NOTEBOOK_FILES[method_name]),
                    "Dashboard Logic": METHOD_REMARKS[method_name],
                }
                for method_name in METHODS
            ]
        )
        st.dataframe(src_df, width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if run:
        st.session_state["req"] = {
            "ds": ds,
            "method": method,
            "k": k,
            "gallery": gallery,
            "queries": queries,
            "train_n": train_n,
            "query_source": query_source,
            "sample_index": sample_index,
            "bytes": uploaded_bytes,
            "label": chosen_label,
        }
    if "req" not in st.session_state:
        st.info("Choose your dataset and query source, then click `Generate Dashboard Results`.")
        return

    req = st.session_state["req"]
    with st.spinner("Generating retrieval results and comparative analysis table..."):
        b = load_bundle(req["ds"], req["gallery"], req["queries"], req["train_n"])
        pack = learned_pack(req["ds"], req["gallery"], req["queries"], req["train_n"])
        qp = query_payload(req, b, pack)
        qd = qp["query_display"]
        gf = gallery_features_for_method(req["method"], pack)
        qf = qp["query_feature"]
        scores = gf @ qf
        idx = np.argsort(-scores)[: req["k"]]
        table = assignment_table(req["ds"], req["gallery"], req["queries"], req["train_n"], req["k"])

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("2. Retrieved Images and Similarity Scores")
    l, r = st.columns([0.92, 1.28])
    with l:
        st.image(qd, width="stretch", caption=f"{qp['caption']} | Label: {qp['label_name']}")
    with r:
        mc_top = st.columns(2)
        with mc_top[0]:
            metric_card("Method", req["method"], "selected retrieval model")
        with mc_top[1]:
            metric_card("Top Match", b["classes"][int(b["gal_y"][idx[0]])], "best retrieved class")
        mc_bottom = st.columns(2)
        with mc_bottom[0]:
            metric_card("Best Similarity", f"{float(scores[idx[0]]):.4f}", "highest cosine score")
        with mc_bottom[1]:
            pred = {}
            for i in idx:
                pred[int(b["gal_y"][i])] = pred.get(int(b["gal_y"][i]), 0.0) + float(scores[i])
            metric_card("Predicted Class", b["classes"][max(pred, key=pred.get)], "weighted by scores")
        if qp["label_idx"] is not None:
            label_idx = int(qp["label_idx"])
            rel = (b["gal_y"][idx] == label_idx).astype(np.int32)
            tot = int((b["gal_y"] == label_idx).sum())
            hits = 0
            ap = 0.0
            dcg = 0.0
            for rank, x in enumerate(rel, 1):
                if x:
                    hits += 1
                    ap += hits / rank
                    dcg += 1.0 / np.log2(rank + 1)
            idcg = sum(1.0 / np.log2(j + 1) for j in range(1, min(tot, len(rel)) + 1))
            qvals = [("Precision", f"{float(rel.sum()/max(req['k'],1)):.4f}"), ("Recall", f"{float(rel.sum()/max(tot,1)):.4f}"), ("AP", f"{float(ap/max(tot,1)):.4f}"), ("Ranking Quality", f"{float(dcg/max(idcg,EPS)):.4f}"), ("Relevant Hits", str(int(rel.sum())))]
            qrow1 = st.columns(3)
            qrow2 = st.columns(2)
            for col, (a, btxt) in zip(qrow1 + qrow2, qvals):
                with col:
                    metric_card(a, btxt)
        else:
            st.info("For uploaded external images, choose the true label if you want query-level Precision, Recall, AP, and Ranking Quality.")

    rows = []
    cols = st.columns(min(5, len(idx)))
    for rank, i in enumerate(idx, 1):
        rows.append({"Rank": rank, "Class": b["classes"][int(b["gal_y"][i])], "Similarity Score": round(float(scores[i]), 4), "Gallery Index": int(i)})
        with cols[(rank - 1) % len(cols)]:
            st.image(b["gal_d"][i], width="stretch")
            st.caption(f"Rank {rank} | {b['classes'][int(b['gal_y'][i])]} | score={float(scores[i]):.4f}")
        if rank % len(cols) == 0 and rank != len(idx):
            cols = st.columns(min(5, len(idx)))
    st.markdown("**Similarity Scores Table**")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("3. Comparative Analysis")
    st.caption(f"Generated on the {req['ds']} held-out benchmark split using Top-{req['k']} retrieval.")
    st.dataframe(table, width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("4. Metric Definitions")
    st.latex(rf"Precision@{req['k']} = \frac{{Relevant\ Retrieved}}{{{req['k']}}}")
    st.latex(r"Recall = \frac{Relevant\ Retrieved}{Total\ Relevant}")
    st.latex(r"mAP = \frac{1}{N}\sum_{q=1}^{N} AP(q)")
    st.latex(r"Robustness = \text{mean mAP under noise, blur, and rotation}")
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
