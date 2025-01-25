import streamlit as st
import cv2
from ultralytics import YOLO
import time
from collections import defaultdict


def video_stream(model, confidence_threshold):
    # เปิดการเชื่อมต่อกับกล้อง (0 หมายถึงกล้องตัวหลัก)
    cap = cv2.VideoCapture(1)  #(กล้อง USB 1, 2, etc.)

    if not cap.isOpened():
        st.error("Cannot access the webcam. Please check your camera connection.")
        return

    # จองพื้นที่ในหน้าเว็บของ Streamlit
    frame_placeholder = st.empty()
    label_count_placeholder = st.empty()

    # สร้างปุ่มสำหรับหยุดการทำงานของกล้อง
    stop_button = st.button("Stop Webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Cannot read frame from webcam.")
            break

        # นำภาพเข้าสู่การประมวลผลโดยโมเดล
        results = model(frame, conf=confidence_threshold)  # ใช้ค่า confidence จากผู้ใช้

        # สร้างตัวแปร Dicationary เพื่อรองรับการนับวัตถุ
        label_counts = defaultdict(int)

        # ตีกรอบและทำ Label ให้วัตถุที่ตรวจจับได้ทั้งหมด
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box

                # ตรวจสอบว่าค่า confidence ผ่านเกณฑ์ที่ตั้งไว้หรือไม่
                if conf >= confidence_threshold:
                    label = f"{model.names[int(cls)]}"
                    label_counts[label] += 1  # Count the label

                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({conf:.2f})", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # แปลงระบบสีจาก BGR เป็น RGB สำหรับแสดงใน Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # อีพเดทตัวเลขใน Label ที่ตรวจจับได้
        label_count_placeholder.markdown("### Object Counts:")
        for label, count in label_counts.items():
            label_count_placeholder.write(f"- **{label}**: {count}")

        # สั้งให้หยุดการทำงานประมวลผลภาพเมื่อปุ่ม Stop ถูกกด
        if stop_button:
            break

        # Optional: Delay to limit frame rate
        time.sleep(0.03)  # ~30 FPS

    # Release the camera when done
    cap.release()

# ฟังก์ชันหลักของโปรแกรม # ใช้โมเดล best.pt 

def main():
    st.title("Real-Time Object Detection")
    st.write("ตัวอย่างโปรแกรมจาก ดร.ไช้ Ignite Innovation")

    # กำหนดไฟล์โมเดล
    model_path = "best.pt"

    # โหลดโมเดล
    st.info("Loading model...")
    model = YOLO(model_path)
    st.success("Model loaded successfully.")

    # สร้าง Slider สำหรับปรับค่า Confidence
    confidence_threshold = st.slider("Set Confidence Threshold", min_value=0.0, max_value=1.0, value=0.99, step=0.01)

    # สร้างปุ่มเริ่มการตรวจจับวัตถุ
    start_button = st.button("Start Webcam")
    if start_button:
        video_stream(model, confidence_threshold)


main()
