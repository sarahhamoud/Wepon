import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import tempfile
from PIL import Image
import cv2
import numpy as np

# =========================
# إعداد الصفحة
# =========================
st.set_page_config(
    page_title="كشف الأسلحة باستخدام YOLO",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CSS: RTL + ألوان فاتحة + خط أوضح + واجهة هندسية
# =========================
st.markdown(
    """
    <style>
    :root{
      --bg:#f7f9fc;
      --card:#ffffff;
      --text:#0f172a;
      --muted:#475569;
      --primary:#2563eb;
      --primary2:#06b6d4;
      --border:#e2e8f0;
      --shadow: 0 10px 30px rgba(2, 6, 23, .08);
    }

    html, body, [data-testid="stApp"]{
        background: var(--bg);
        direction: RTL;
        text-align: right;
        font-family: "Segoe UI", Tahoma, Arial, sans-serif;
        color: var(--text);
    }

    /* إزالة الهيدر العلوي الافتراضي (إذا يظهر) */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    h1,h2,h3,h4,h5,h6,p,label,span,div{
        text-align: right !important;
        color: var(--text);
    }

    /* تكبير الخط العام */
    .stMarkdown, .stTextInput, .stSelectbox, .stRadio, .stButton button{
        font-size: 16px !important;
    }

    /* بطاقات */
    .card{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 18px;
        box-shadow: var(--shadow);
        padding: 16px 18px;
        margin-bottom: 16px;
    }

    /* عنوان علوي */
    .topbar{
        display:flex;
        justify-content: space-between;
        align-items:center;
        gap: 12px;
        padding: 14px 16px;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(37,99,235,.08), rgba(6,182,212,.10));
        border: 1px solid rgba(37,99,235,.18);
        box-shadow: var(--shadow);
        margin-bottom: 16px;
    }
    .brand{
        font-weight: 700;
        font-size: 22px;
        color: var(--text);
        margin:0;
    }
    .owner{
        font-weight: 600;
        font-size: 14px;
        color: var(--muted);
        margin:0;
        direction:ltr;
        text-align:left !important;
    }

    /* الأزرار */
    .stButton button{
        border-radius: 14px !important;
        border: 1px solid rgba(37,99,235,.25) !important;
        padding: 12px 14px !important;
        font-weight: 700 !important;
        background: #ffffff !important;
        color: var(--text) !important;
        box-shadow: 0 8px 22px rgba(2, 6, 23, .06) !important;
        transition: all .2s ease;
    }
    .stButton button:hover{
        transform: translateY(-1px);
        border-color: rgba(37,99,235,.45) !important;
        box-shadow: 0 14px 30px rgba(2, 6, 23, .10) !important;
    }

    /* الراديو/الاختيارات */
    [data-testid="stRadio"]{
        background: #fff;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 12px 14px;
        box-shadow: 0 10px 25px rgba(2, 6, 23, .06);
    }

    /* مدخلات */
    .stTextInput input, .stSelectbox div[data-baseweb="select"]{
        border-radius: 14px !important;
    }

    /* تنسيق رسائل الحالة */
    .stAlert{
        border-radius: 14px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# شريط علوي (بدون “بيئة عربية”)
# =========================
st.markdown(
    """
    <div class="topbar">
      <p class="brand">تطبيق كشف الأسلحة باستخدام YOLO</p>
      <p class="owner">sarah hamoud hussien</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="card">
      <p style="margin:0;color:#334155;font-weight:600;">
      اختاري وضع العمل: <b>صورة</b> أو <b>فيديو</b> أو <b>كاميرا الهاتف/الحاسوب</b>.
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# مسارات المشروع (مناسبة للموبايل/السيرفر)
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
OUTPUT_DIR = BASE_DIR / "outputs" / "video_inference"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# تحميل الموديل مرة واحدة
# =========================
@st.cache_resource
def load_model(path: Path):
    return YOLO(str(path))

if not MODEL_PATH.exists():
    st.error("ملف النموذج غير موجود! ضعي best.pt داخل مجلد: models/")
    st.stop()

model = load_model(MODEL_PATH)

# =========================
# رسم جميع المربعات
# =========================
def draw_boxes(frame_bgr, results):
    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            label = f"{model.names.get(cls, 'obj')} {conf:.2f}"

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 180, 0), 2)
            cv2.putText(frame_bgr, label, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)

    return frame_bgr

# =========================
# معالجة فيديو كامل مع تقدم
# =========================
def process_video(input_path: str, output_path: Path, conf=0.20, iou=0.40, imgsz=640):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("❌ لا يمكن فتح ملف الفيديو")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    prog = st.progress(0)
    txt = st.empty()

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = model(frame, imgsz=imgsz, conf=conf, iou=iou)
        annotated = draw_boxes(frame, results)
        out.write(annotated)

        if total > 0:
            prog.progress(min(frame_count / total, 1.0))
            txt.write(f"جاري المعالجة: {frame_count} / {total} إطار")

    cap.release()
    out.release()
    prog.progress(1.0)
    txt.write("✅ اكتملت المعالجة")

# =========================
# إعدادات الكشف
# =========================
with st.sidebar:
    st.markdown("### إعدادات الكشف")
    conf_th = st.slider("عتبة الثقة (Confidence)", 0.05, 0.90, 0.20, 0.05)
    iou_th = st.slider("عتبة التداخل (IoU)", 0.05, 0.90, 0.40, 0.05)
    img_size = st.select_slider("حجم الإدخال (imgsz)", options=[320, 416, 512, 640, 768], value=640)
    st.markdown("---")
    st.markdown("### ملاحظة للموبايل")
    st.caption("وضع الكاميرا يعمل على الهاتف عبر التقاط صورة واحدة. بث مباشر غير مدعوم في Streamlit.")

# =========================
# وضع الصورة
# =========================
def run_image_mode():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🖼️ وضع الصورة")
    uploaded_file = st.file_uploader("ارفع/ارفعي صورة", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="الصورة الأصلية", use_container_width=True)

        with st.spinner("جاري الكشف..."):
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            results = model(img_bgr, imgsz=img_size, conf=conf_th, iou=iou_th)

            annotated_bgr = draw_boxes(img_bgr.copy(), results)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        st.success("تم الكشف ✅")
        st.image(annotated_rgb, caption="النتيجة", use_container_width=True)

# =========================
# وضع الفيديو
# =========================
def run_video_mode():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎬 وضع الفيديو")
    video_file = st.file_uploader("ارفع/ارفعي فيديو", type=["mp4", "avi", "mov", "mkv"])
    st.markdown("</div>", unsafe_allow_html=True)

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.flush()

        st.video(tfile.name)

        colA, colB = st.columns([1, 1])
        with colA:
            start = st.button("🚀 بدء المعالجة", use_container_width=True)
        with colB:
            st.caption("نصيحة: فيديوهات قصيرة أفضل للموبايل.")

        if start:
            st.info("جاري المعالجة... قد يستغرق ذلك حسب طول الفيديو.")
            output_path = OUTPUT_DIR / (Path(video_file.name).stem + "_processed.mp4")

            process_video(tfile.name, output_path, conf=conf_th, iou=iou_th, imgsz=img_size)

            st.success("تمت معالجة الفيديو بنجاح ✅")

            st.markdown("#### تحميل الفيديو المعالج")
            with open(output_path, "rb") as f:
                st.download_button(
                    label="⬇️ تنزيل الفيديو بعد المعالجة",
                    data=f.read(),
                    file_name=output_path.name,
                    mime="video/mp4",
                    use_container_width=True
                )

# =========================
# وضع الكاميرا (موبايل/حاسوب)
# =========================
def run_camera_mode():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📸 وضع الكاميرا (التقاط صورة)")
    st.write("على الهاتف: سيفتح الكاميرا مباشرة. على الحاسوب: يستخدم كاميرا اللابتوب إن وُجدت.")
    st.markdown("</div>", unsafe_allow_html=True)

    img_data = st.camera_input("التقط/التقطي صورة")

    if img_data is not None:
        image = Image.open(img_data).convert("RGB")
        st.image(image, caption="الصورة الملتقطة", use_container_width=True)

        with st.spinner("جاري الكشف..."):
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            results = model(img_bgr, imgsz=img_size, conf=conf_th, iou=iou_th)

            annotated_bgr = draw_boxes(img_bgr.copy(), results)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        st.success("تم الكشف ✅")
        st.image(annotated_rgb, caption="النتيجة", use_container_width=True)

# =========================
# واجهة الأوضاع (أزرار كبيرة)
# =========================
if "mode" not in st.session_state:
    st.session_state["mode"] = "image"

cols = st.columns(3)
with cols[0]:
    if st.button("🖼️ صورة", use_container_width=True):
        st.session_state["mode"] = "image"
with cols[1]:
    if st.button("🎬 فيديو", use_container_width=True):
        st.session_state["mode"] = "video"
with cols[2]:
    if st.button("📸 كاميرا", use_container_width=True):
        st.session_state["mode"] = "camera"

st.markdown("---")

mode = st.session_state["mode"]
if mode == "image":
    run_image_mode()
elif mode == "video":
    run_video_mode()
elif mode == "camera":
    run_camera_mode()
