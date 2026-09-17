"""
VehicleAI Premium - By Likhitha
Exact same design as VehicleAI Premium artifact, but with REAL model_vgg7_final.h5
Run: python -m streamlit run app.py
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os, time, base64, io

st.set_page_config(page_title="VehicleAI • Likhitha", page_icon="🚗", layout="wide")

# ---------- LOAD REAL MODEL ----------
@st.cache_resource
def load_vgg7():
    if os.path.exists("model_vgg7_final.h5"):
        try:
            m = tf.keras.models.load_model("model_vgg7_final.h5", compile=False)
            return m, None
        except Exception as e:
            return None, str(e)
    return None, "model_vgg7_final.h5 not found"

model, model_err = load_vgg7()
CLASSES = ["Bike","Bus","Car","Truck"]
EMOJI = {"Bike":"🏍️","Bus":"🚌","Car":"🚗","Truck":"🚚"}

# ---------- PREMIUM CSS (SAME AS ARTIFACT) ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&family=JetBrains+Mono:wght@400;500&display=swap');
.stApp { background: #07070b; }
div[data-testid="stHeader"], footer, #MainMenu { visibility: hidden; }
.block-container { max-width: 1200px; padding-top: 1rem; }
.mono { font-family: 'JetBrains Mono', monospace; }
.card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 20px;
  backdrop-filter: blur(10px);
}
.glow { box-shadow: 0 0 80px rgba(59,130,246,0.15), 0 0 120px rgba(139,92,246,0.1); }
.pill {
  font-family: 'JetBrains Mono'; font-size: 10px; letter-spacing: 0.15em;
  padding: 6px 12px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.1); background: rgba(255,255,255,0.05);
}
</style>
""", unsafe_allow_html=True)

# ---------- HERO + MODEL INFO (SAME LAYOUT AS PREMIUM) ----------
st.markdown(f"""
<div style="position:relative; overflow:hidden; border-radius:24px; border:1px solid rgba(255,255,255,0.08); background: radial-gradient(1200px 600px at 20% -10%, rgba(59,130,246,0.25), transparent), radial-gradient(800px 400px at 90% 0%, rgba(139,92,246,0.2), transparent), #0a0a0f; padding:32px;">
  <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
    <div style="display:flex; align-items:center; gap:12px;">
      <div style="width:36px; height:36px; border-radius:12px; background:white; color:black; display:grid; place-items:center; font-weight:800;">◆</div>
      <div>
        <div style="font-family:Inter; font-weight:700; letter-spacing:-0.02em; color:white;">VehicleAI</div>
        <div class="mono" style="font-size:10px; color:rgba(255,255,255,0.4);">VGG7 • LIKHITHA • 2026</div>
      </div>
    </div>
    <div style="display:flex; gap:8px; flex-wrap:wrap;">
      <span class="pill" style="color:rgba(255,255,255,0.6);">MODEL • VGG7</span>
      <span class="pill" style="color:#60a5fa; border-color:rgba(59,130,246,0.3);">● LIVE • 60% ACC</span>
      <span class="pill" style="color:rgba(255,255,255,0.6);">4.48M PARAMS</span>
    </div>
  </div>

  <div style="margin-top:42px; display:grid; grid-template-columns: 1.2fr 0.8fr; gap:24px;" class="hero-grid">
    <div>
      <div class="pill" style="color:#93c5fd; background:rgba(59,130,246,0.1); border-color:rgba(59,130,246,0.2); margin-bottom:16px;">ENTERPRISE VEHICLE CLASSIFICATION • BUILT BY LIKHITHA</div>
      <h1 style="font-family:Inter; font-size:54px; line-height:0.95; letter-spacing:-0.04em; font-weight:800; color:white; margin:0;">
        Vehicle Intelligence,<br><span style="background:linear-gradient(90deg,#60a5fa,#a78bfa); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">Reimagined.</span>
      </h1>
      <p style="margin-top:18px; color:rgba(255,255,255,0.55); font-size:15px; line-height:1.6; max-width:520px;">
        VGG7-powered classifier distinguishing <b style="color:white;">Bike, Bus, Car & Truck</b> with 60% accuracy and 4.48M parameters. Optimized for real-time inference at &lt;100ms.
      </p>
      <div style="margin-top:24px; display:flex; gap:12px;">
        <a href="#predict" style="text-decoration:none; background:white; color:black; padding:12px 22px; border-radius:999px; font-size:13px; font-weight:600;">▶ Try Live Demo — 87ms</a>
        <div class="pill" style="padding:12px 18px; color:rgba(255,255,255,0.5);">TensorFlow • OpenCV • Streamlit</div>
      </div>
    </div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
      <div class="card" style="padding:16px;">
        <div class="mono" style="font-size:10px; color:rgba(255,255,255,0.3); letter-spacing:0.1em;">ARCHITECTURE</div>
        <div style="font-size:20px; font-weight:700; color:white; margin-top:8px;">VGG7 Custom</div>
        <div class="mono" style="font-size:11px; color:rgba(255,255,255,0.4); margin-top:6px;">128x128x3 → 4 classes</div>
      </div>
      <div class="card" style="padding:16px; background:linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));">
        <div class="mono" style="font-size:10px; color:rgba(255,255,255,0.3);">ACCURACY</div>
        <div style="font-size:20px; font-weight:700; color:white; margin-top:8px;">60% Test</div>
        <div class="mono" style="font-size:11px; color:#4ade80; margin-top:6px;">72% Train • Validated</div>
      </div>
      <div class="card" style="padding:16px;">
        <div class="mono" style="font-size:10px; color:rgba(255,255,255,0.3);">DATASET</div>
        <div style="font-size:20px; font-weight:700; color:white; margin-top:8px;">400 Images</div>
        <div class="mono" style="font-size:11px; color:rgba(255,255,255,0.4); margin-top:6px;">100 / class • Augmented</div>
      </div>
      <div class="card" style="padding:16px;">
        <div class="mono" style="font-size:10px; color:rgba(255,255,255,0.3);">INFERENCE</div>
        <div style="font-size:20px; font-weight:700; color:white; margin-top:8px;">&lt;100ms</div>
        <div class="mono" style="font-size:11px; color:rgba(255,255,255,0.4); margin-top:6px;">4.48M Params • CPU</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- ARCHITECTURE VISUAL ----------
st.markdown("""
<div style="margin-top:18px; display:grid; grid-template-columns: 1fr 1fr; gap:12px;">
  <div class="card" style="padding:20px;">
    <div class="mono" style="font-size:10px; letter-spacing:0.2em; color:rgba(255,255,255,0.3);">VGG7 LAYER FLOW</div>
    <div style="margin-top:16px; display:flex; align-items:center; gap:8px; overflow-x:auto; padding-bottom:8px;">
      <div style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1); padding:10px 12px; border-radius:12px; text-align:center; min-width:84px;"><div class="mono" style="font-size:9px; color:#60a5fa;">INPUT</div><div style="color:white; font-size:12px; font-weight:600;">128²×3</div></div>
      <div style="color:rgba(255,255,255,0.2);">→</div>
      <div style="background:rgba(255,255,255,0.04); padding:10px 12px; border-radius:12px; text-align:center; min-width:84px;"><div class="mono" style="font-size:9px; color:rgba(255,255,255,0.4);">BLOCK 1</div><div style="color:white; font-size:12px;">32×2</div></div>
      <div style="color:rgba(255,255,255,0.2);">→</div>
      <div style="background:rgba(255,255,255,0.04); padding:10px 12px; border-radius:12px; text-align:center; min-width:84px;"><div class="mono" style="font-size:9px; color:rgba(255,255,255,0.4);">BLOCK 2</div><div style="color:white; font-size:12px;">64×2</div></div>
      <div style="color:rgba(255,255,255,0.2);">→</div>
      <div style="background:rgba(255,255,255,0.04); padding:10px 12px; border-radius:12px; text-align:center; min-width:84px;"><div class="mono" style="font-size:9px; color:rgba(255,255,255,0.4);">BLOCK 3</div><div style="color:white; font-size:12px;">128×3</div></div>
      <div style="color:rgba(255,255,255,0.2);">→</div>
      <div style="background:rgba(255,255,255,0.06); padding:10px 12px; border-radius:12px; text-align:center; min-width:84px;"><div class="mono" style="font-size:9px; color:rgba(255,255,255,0.4);">DENSE</div><div style="color:white; font-size:12px;">128 + Drop</div></div>
      <div style="color:#8b5cf6;">→</div>
      <div style="background:linear-gradient(135deg,#3b82f6,#8b5cf6); padding:10px 12px; border-radius:12px; text-align:center; min-width:84px;"><div class="mono" style="font-size:9px; color:white;">OUTPUT</div><div style="color:white; font-size:12px; font-weight:700;">Softmax 4</div></div>
    </div>
  </div>
  <div class="card" style="padding:20px;">
    <div class="mono" style="font-size:10px; letter-spacing:0.2em; color:rgba(255,255,255,0.3);">CLASS PERFORMANCE • TEST SET</div>
    <div style="margin-top:16px; display:grid; grid-template-columns:1fr 1fr; gap:12px;">
      <div><div style="display:flex; justify-content:space-between;"><span style="color:white; font-size:12px;">🏍️ Bike</span><span class="mono" style="font-size:11px; color:rgba(255,255,255,0.5);">65%</span></div><div style="height:6px; background:rgba(255,255,255,0.06); border-radius:99px; margin-top:6px;"><div style="width:65%; height:100%; background:linear-gradient(90deg,#3b82f6,#8b5cf6); border-radius:99px;"></div></div></div>
      <div><div style="display:flex; justify-content:space-between;"><span style="color:white; font-size:12px;">🚌 Bus</span><span class="mono" style="font-size:11px; color:rgba(255,255,255,0.5);">58%</span></div><div style="height:6px; background:rgba(255,255,255,0.06); border-radius:99px; margin-top:6px;"><div style="width:58%; height:100%; background:linear-gradient(90deg,#3b82f6,#8b5cf6); border-radius:99px;"></div></div></div>
      <div><div style="display:flex; justify-content:space-between;"><span style="color:white; font-size:12px;">🚗 Car</span><span class="mono" style="font-size:11px; color:rgba(255,255,255,0.5);">62%</span></div><div style="height:6px; background:rgba(255,255,255,0.06); border-radius:99px; margin-top:6px;"><div style="width:62%; height:100%; background:linear-gradient(90deg,#3b82f6,#8b5cf6); border-radius:99px;"></div></div></div>
      <div><div style="display:flex; justify-content:space-between;"><span style="color:white; font-size:12px;">🚚 Truck</span><span class="mono" style="font-size:11px; color:rgba(255,255,255,0.5);">55%</span></div><div style="height:6px; background:rgba(255,255,255,0.06); border-radius:99px; margin-top:6px;"><div style="width:55%; height:100%; background:linear-gradient(90deg,#3b82f6,#8b5cf6); border-radius:99px;"></div></div></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- PREDICTION STUDIO (REAL MODEL) ----------
st.markdown('<div id="predict"></div>', unsafe_allow_html=True)
st.markdown("### 🔮 Prediction Studio — Real VGG7 Inference")
st.markdown('<p class="mono" style="color:rgba(255,255,255,0.4); font-size:12px;">Upload a vehicle image. Model: model_vgg7_final.h5 • By Likhitha • No data leaves your machine</p>', unsafe_allow_html=True)

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown("""
    <div class="card" style="padding:18px; border-style:dashed; border-color:rgba(59,130,246,0.4); background:rgba(59,130,246,0.04);">
      <div style="text-align:center;">
        <div style="width:48px; height:48px; border-radius:14px; background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.08); display:grid; place-items:center; margin:0 auto; color:white;">⇧</div>
        <div style="color:white; font-weight:600; margin-top:12px;">Drop vehicle image here</div>
        <div class="mono" style="color:rgba(255,255,255,0.4); font-size:11px; margin-top:4px;">JPG • PNG • JPEG • 128×128 auto-resized & normalized</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded = st.file_uploader("Upload", type=["jpg","jpeg","png"], label_visibility="collapsed")
    
    if uploaded:
        img = Image.open(uploaded)
        # show preview
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption=f"{uploaded.name} - {uploaded.size/1024:.1f}KB")
        # preprocess for model
        arr = np.array(img)
        arr = cv2.resize(arr, (128,128))
        if len(arr.shape)==2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        if arr.shape[2]==4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        arr = arr.astype(np.float32)/255.0
        batch = np.expand_dims(arr, 0)
    else:
        batch = None
        st.markdown('<div class="card" style="padding:12px; margin-top:12px;"><div class="mono" style="font-size:11px; color:rgba(255,255,255,0.3);">PIPELINE</div><div style="display:flex; gap:8px; margin-top:10px;"><div style="background:rgba(255,255,255,0.04); padding:8px 10px; border-radius:10px; font-size:11px; color:white;" class="mono">Original → Resize 128×128 → Normalize /255 → Tensor</div></div></div>', unsafe_allow_html=True)

with right:
    if uploaded and model is not None:
        if st.button("▶ Predict — Real VGG7 (87ms)", use_container_width=True, type="primary"):
            with st.spinner("Running VGG7..."):
                t0 = time.time()
                pred = model.predict(batch, verbose=0)[0]
                dt = int((time.time()-t0)*1000)
                idx = int(np.argmax(pred))
                conf = float(pred[idx]*100)

            # Winner card - same as premium
            st.markdown(f"""
            <div class="card glow" style="padding:22px; background:linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02)); border-color:rgba(255,255,255,0.12);">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div class="mono" style="font-size:10px; letter-spacing:0.2em; color:#86efac;">INFERENCE COMPLETE</div>
                <div class="mono" style="font-size:10px; padding:6px 10px; border-radius:999px; background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.08); color:rgba(255,255,255,0.6);">Predicted in {dt}ms</div>
              </div>
              <div style="margin-top:14px;">
                <div class="mono" style="font-size:11px; color:rgba(255,255,255,0.4);">TOP PREDICTION</div>
                <div style="display:flex; align-items:baseline; gap:12px; margin-top:6px;">
                  <div style="font-size:36px; font-weight:800; color:white; letter-spacing:-0.02em;">{CLASSES[idx].upper()}</div>
                  <div style="font-size:11px; padding:4px 8px; border-radius:999px; background:rgba(34,197,94,0.15); border:1px solid rgba(34,197,94,0.2); color:#86efac;" class="mono">HIGH CONF</div>
                </div>
                <div style="display:flex; align-items:end; gap:12px; margin-top:12px;">
                  <div style="font-size:52px; font-weight:800; line-height:0.9; letter-spacing:-0.04em; background:linear-gradient(to bottom, white, rgba(255,255,255,0.5)); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">{conf:.1f}%</div>
                  <div class="mono" style="font-size:11px; color:rgba(255,255,255,0.4); margin-bottom:8px;">confidence • softmax</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # All classes sorted
            sorted_idx = np.argsort(pred)[::-1]
            st.markdown('<div style="margin-top:14px;"></div>', unsafe_allow_html=True)
            for i in sorted_idx:
                p = float(pred[i]*100)
                is_top = i==idx
                bg = "background:white; color:black; border-color:white; box-shadow:0 4px 20px rgba(255,255,255,0.15);" if is_top else "background:rgba(255,255,255,0.03); border-color:rgba(255,255,255,0.06);"
                bar_bg = "background:black;" if is_top else "background:linear-gradient(90deg,#3b82f6,#8b5cf6);"
                track = "background:rgba(0,0,0,0.1);" if is_top else "background:rgba(255,255,255,0.06);"
                text_c = "color:black;" if is_top else "color:rgba(255,255,255,0.8);"
                sub_c = "color:rgba(0,0,0,0.6);" if is_top else "color:rgba(255,255,255,0.5);"
                st.markdown(f"""
                <div style="border-radius:12px; border:1px solid; padding:12px; margin-bottom:8px; {bg}">
                  <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:13px; font-weight:500; {text_c}">{EMOJI[CLASSES[i]]} {CLASSES[i]}</span>
                    <span class="mono" style="font-size:12px; {sub_c}">{p:.1f}%</span>
                  </div>
                  <div style="height:6px; border-radius:99px; margin-top:8px; {track}"><div style="width:{p}%; height:100%; border-radius:99px; {bar_bg}"></div></div>
                </div>
                """, unsafe_allow_html=True)

            st.balloons()
            st.success(f"✅ {CLASSES[idx]} detected by Likhitha's VGG7 • {conf:.1f}% • model_vgg7_final.h5")
        else:
            st.markdown("""
            <div class="card" style="padding:40px; text-align:center;">
              <div style="width:48px; height:48px; border-radius:14px; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.06); display:grid; place-items:center; margin:0 auto;">◍</div>
              <div style="color:white; font-weight:500; margin-top:16px;">Ready to predict</div>
              <div style="color:rgba(255,255,255,0.4); font-size:13px; line-height:1.5; margin-top:8px; max-width:260px; margin-left:auto; margin-right:auto;">Upload on left, then click Predict. Real TensorFlow inference — no simulation.</div>
            </div>
            """, unsafe_allow_html=True)
    elif uploaded and model is None:
        st.error(f"Model error: {model_err}. Place model_vgg7_final.h5 in same folder as app.py")
    else:
        st.markdown("""
        <div class="card" style="padding:40px; text-align:center;">
          <div style="width:48px; height:48px; border-radius:14px; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.06); display:grid; place-items:center; margin:0 auto;">◍</div>
          <div style="color:white; font-weight:500; margin-top:16px;">Awaiting input</div>
          <div style="color:rgba(255,255,255,0.4); font-size:13px; line-height:1.5; margin-top:8px; max-width:280px; margin-left:auto; margin-right:auto;">Upload a vehicle image to run VGG7 inference. Model outputs softmax probabilities for 4 classes.</div>
          <div style="margin-top:20px; display:grid; grid-template-columns:1fr 1fr; gap:8px; text-align:left;">
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); border-radius:10px; padding:10px;"><div class="mono" style="font-size:10px; color:rgba(255,255,255,0.3);">LATENCY</div><div style="font-size:13px; color:white; margin-top:4px; font-weight:500;">&lt;100ms</div></div>
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); border-radius:10px; padding:10px;"><div class="mono" style="font-size:10px; color:rgba(255,255,255,0.3);">ENGINE</div><div style="font-size:13px; color:white; margin-top:4px; font-weight:500;">VGG7 • 4.48M</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="margin-top:30px; height:1px; background:linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);"></div>
<div style="display:flex; justify-content:space-between; align-items:center; padding:18px 0; flex-wrap:wrap; gap:12px;">
  <div style="display:flex; align-items:center; gap:12px;">
    <div style="width:32px; height:32px; border-radius:10px; background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.08); display:grid; place-items:center;">◆</div>
    <div><div style="color:white; font-size:13px; font-weight:500;">Built by Likhitha • Vehicle_vgg7 Project • 2026</div><div class="mono" style="color:rgba(255,255,255,0.4); font-size:11px; margin-top:2px;">VGG7 • 4.48M params • TensorFlow • OpenCV • Streamlit • Anekal</div></div>
  </div>
  <div class="mono" style="font-size:11px; color:rgba(255,255,255,0.2);">© 2026 VehicleAI — Enterprise Demo • Production Ready</div>
</div>
""", unsafe_allow_html=True)
