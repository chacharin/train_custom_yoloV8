# นำเข้าไลบรารีที่จำเป็น
import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
from collections import defaultdict

# ฟังก์ชันหลักสำหรับการแสดงผลผ่าน Streamlit
def main():
    st.title("Object Detection")
    st.write("ตัวอย่างโปรแกรมจาก ดร.ไช้ Ignite Innovation")

    # กำหนดเส้นทางของโมเดล YOLOv8
    model_path = "best.pt"  # เปลี่ยนเป็นชื่อไฟล์โมเดลของเรา

    # ส่วนสำหรับอัปโหลดไฟล์ภาพ
    uploaded_file = st.file_uploader("เลือกไฟล์ภาพ...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # บันทึกไฟล์ที่อัปโหลดลงในโฟลเดอร์ชั่วคราว
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tfile.write(uploaded_file.read())

        # โหลดโมเดล YOLO
        st.info("Model Loading ...")
        model = YOLO(model_path) 

        # อ่านไฟล์ภาพที่อัปโหลดด้วย OpenCV
        img = cv2.imread(tfile.name)
        
        # แปลงสีภาพเป็นระบบ RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

        # เริ่มการตรวจจับวัตถุด้วยโมเดล
        st.info("Object Detecting ...")
        results = model(img)

        # เตรียมตัวแปรสำหรับการเก็บข้อมูล
        label_count = defaultdict(int)

        # ดึงข้อมูลและวาดกรอบ Bounding Box
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box
                label = f"{model.names[int(cls)]}"
                label_count[label] += 1

                # วาดกรอบสี่เหลี่ยมและใส่ข้อความชื่อวัตถุลงบนภาพ
                cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img_rgb, f"{label} ({conf:.2f})", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # แสดงภาพผลลัพธ์
        st.image(img_rgb, caption="ผลการทำ Object Detection", use_container_width=True)

        # แสดงจำนวนวัตถุแยกตามประเภท
        st.subheader("สรุปผลการ Detect")
        for label, count in label_count.items():
            st.write(f"- **{label}**: {count}")

        # แจ้งสถานะการทำงานสำเร็จ
        st.success("Object Detection Completed")

        # ลบไฟล์ชั่วคราว
        try:
            tfile.close()  # ปิดไฟล์ก่อนลบ
            os.unlink(tfile.name)  # ลบไฟล์
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการลบไฟล์ชั่วคราว: {e}")

main()
