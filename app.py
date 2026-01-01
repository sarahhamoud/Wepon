import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import tempfile
from PIL import Image
import cv2
import numpy as np
import os

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("يجب تسجيل الدخول من الصفحة الرئيسية أولاً.")
    st.stop()

# ---------------- إعداد الصفحة ----------------
st.set_page_config(
    page_title="تطبيق كشف الأسلحة باستخدام YOLO",
    layout="wide"
)

# ✅ تفعيل اتجاه من اليمين لليسار + محاذاة النص إلى اليمين
st.markdown(
    """
    <style>
    html, body, [data-testid="stApp"] {
        direction: RTL;
        text-align: right;
    }
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔫 تطبيق كشف الأسلحة باستخدام YOLO")
st.write("اختاري أحد الأوضاع الثلاثة: صورة، فيديو، أو كاميرا الحاسوب.")

# ---------------- تحميل الموديل مرة واحدة فقط ----------------
MODEL_PATH = r"C:\Users\hp\Desktop\YOLO_project_backup\train2\weights\best.pt"

@st.cache_resource
def load_model(path):
    model = YOLO(path)
    return model

model = load_model(MODEL_PATH)

# ✅ مجلد حفظ الفيديوهات المعالجة
OUTPUT_DIR = r"C:\Users\hp\Desktop\YOLO_project_backup\runs\streamlit\video_inference"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # يتأكد أن المجلد موجود

# ---------------- دالة ترسم جميع المربعات ----------------
def draw_boxes(frame, results):
    """
    ترسم كل الـ bounding boxes على الفريم.
    frame: صورة BGR (من OpenCV)
    results: ناتج model(...)
    """
    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            # إحداثيات الصندوق
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # نسبة الثقة والفئة
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            label = f"{model.names.get(cls, 'obj')} {conf:.2f}"

            # رسم المستطيل + النص
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

# ---------------- دالة معالجة فيديو كامل وإخراج فيديو جديد ----------------
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("❌ لا يمكن فتح ملف الفيديو")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 🔥 تطبيق YOLO على كل فريم
        results = model(
            frame,
            imgsz=640,
            conf=0.20,  # أقل شوية عشان يكشف أسلحة أكثر
            iou=0.40    # يقلل دمج الصناديق القريبة
        )

        annotated_frame = draw_boxes(frame, results)
        out.write(annotated_frame)

    cap.release()
    out.release()

# ---------------- وضع الصورة ----------------
def run_image_mode():
    st.subheader("📷 وضع الصورة")
    uploaded_file = st.file_uploader("ارفعي صورة", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="الصورة الأصلية", use_container_width=True)

        with st.spinner("جاري الكشف عن الأسلحة في الصورة..."):
            # تحويل الصورة لـ NumPy ثم BGR
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # 🔥 نستخدم model(...) مع conf و iou أقل شوية
            results = model(
                img_bgr,
                imgsz=640,
                conf=0.20,
                iou=0.40
            )

            annotated_bgr = draw_boxes(img_bgr.copy(), results)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        st.success("تمت عملية الكشف ✅")
        st.image(annotated_rgb, caption="نتيجة الكشف (كل المربعات)", use_container_width=True)

# ---------------- وضع الفيديو ----------------
def run_video_mode():
    st.subheader("🎞️ وضع الفيديو")
    video_file = st.file_uploader("ارفعي فيديو", type=["mp4", "avi", "mov", "mkv"])

    if video_file is not None:
        # حفظ الفيديو المرفوع في ملف مؤقت
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.flush()

        # عرض الفيديو الأصلي
        st.video(tfile.name)

        if st.button("🚀 بدء المعالجة"):
            st.info("جاري المعالجة...")

            # مسار حفظ الفيديو الناتج
            output_path = os.path.join(
                OUTPUT_DIR,
                Path(video_file.name).stem + "_processed.mp4"
            )

            # معالجة الفيديو
            process_video(tfile.name, output_path)

            st.success("تمت معالجة الفيديو بنجاح!")


            # عرض المسار
            st.write("مسار الملف المعالج")
            st.code(output_path)

            # زر تحميل
            with open(output_path, "rb") as f:
                st.download_button(
                    label="⬇️ تحميل الفيديو بعد المعالجة",
                    data=f.read(),
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )


# ---------------- وضع الكاميرا ----------------
def run_camera_mode():
    st.subheader("📸 وضع كاميرا الحاسوب (صورة واحدة)")
    st.write("اضغطي على زر الكاميرا لالتقاط صورة من كاميرا اللابتوب، ثم سأطبق عليها نموذج YOLO.")

    img_data = st.camera_input("التقطي صورة من الكاميرا")

    if img_data is not None:
        image = Image.open(img_data).convert("RGB")
        st.image(image, caption="الصورة الملتقطة", use_container_width=True)

        with st.spinner("جاري الكشف عن الأسلحة في الصورة..."):
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            results = model(
                img_bgr,
                imgsz=640,
                conf=0.20,
                iou=0.40
            )

            annotated_bgr = draw_boxes(img_bgr.copy(), results)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        st.success("تمت عملية الكشف ✅")
        st.image(annotated_rgb, caption="نتيجة الكشف (كل المربعات)", use_container_width=True)

# ---------------- واجهة الأوضاع الثلاثة ----------------
if "mode" not in st.session_state:
    st.session_state["mode"] = "image"

cols = st.columns(3)

with cols[0]:
    if st.button("🖼️ كشف من صورة", use_container_width=True):
        st.session_state["mode"] = "image"

with cols[1]:
    if st.button("🎬 كشف من فيديو", use_container_width=True):
        st.session_state["mode"] = "video"

with cols[2]:
    if st.button("📹 كشف من كاميرا الحاسوب", use_container_width=True):
        st.session_state["mode"] = "camera"

st.markdown("---")

mode = st.session_state["mode"]

if mode == "image":
    run_image_mode()
elif mode == "video":
    run_video_mode()
elif mode == "camera":
    run_camera_mode()
