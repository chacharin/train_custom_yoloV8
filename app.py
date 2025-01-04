# นำเข้าไลบรารีที่จำเป็น
import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
from collections import defaultdict
import numpy as np  # สำหรับการคำนวณที่เกี่ยวข้องกับ NMS

# ฟังก์ชันสำหรับ Non-Maximum Suppression (NMS)
def non_max_suppression(boxes, scores, iou_threshold):
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes, scores=scores, score_threshold=0.6, nms_threshold=iou_threshold
    )
    return indices.ravel().tolist() if indices is not None and len(indices) > 0 else []

# ฟังก์ชันหลักสำหรับการแสดงผลผ่าน Streamlit
def main():
    st.title("โปรแกรม Object Detection")
    st.write("เริ่มใช้งานโดยการอัปโหลดไฟล์ภาพ")

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

        # เตรียมตัวแปรสำหรับการเก็บพิกัด Bounding Box และค่าความเชื่อมั่น
        boxes = []
        confidences = []
        detection_info = []
        label_count = defaultdict(int)

        # ดึงข้อมูล Bounding Box และค่าความเชื่อมั่นจากผลลัพธ์
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
                confidences.append(float(conf))

        # เรียกใช้ฟังก์ชัน NMS
        selected_indices = non_max_suppression(boxes, confidences, iou_threshold=1)

        # วาดกรอบและแสดงผลเฉพาะ Bounding Box ที่ผ่าน NMS แล้ว
        for idx in selected_indices:
            x1, y1, x2, y2 = boxes[idx]
            confidence = confidences[idx]
            label = f"{model.names[int(results[0].boxes.data[idx][-1])]}"  # ระบุชื่อวัตถุ (Label)

            # นับจำนวนวัตถุแยกตามประเภท
            label_count[label] += 1

            # เก็บข้อมูล Bounding Box ที่ตรวจจับได้
            detection_info.append({"Label": label, "Confidence": f"{confidence:.2f}", "X1": x1, "Y1": y1, "X2": x2, "Y2": y2})

            # วาดกรอบสี่เหลี่ยมและใส่ข้อความชื่อวัตถุลงบนภาพ
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(img_rgb, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # แสดงภาพผลลัพธ์
        st.image(img_rgb, caption="ผลการทำ Object Detection", use_container_width=True)

        # แสดงจำนวนวัตถุแยกตามประเภท
        st.subheader("สรุปผลการ Detect")
        for label, count in label_count.items():
            st.write(f"- **{label}**: {count}")

        # แสดงรายละเอียดของ Bounding Box
        st.subheader("รายละเอียดกรอบ Bounding Box ")
        for info in detection_info:
            st.write(f"**ชื่อวัตถุ**: {info['Label']} | **ความเชื่อมั่น**: {info['Confidence']} | "
                     f"**ตำแหน่ง**: (X1: {info['X1']}, Y1: {info['Y1']}, X2: {info['X2']}, Y2: {info['Y2']})")

        # แจ้งสถานะการทำงานสำเร็จ
        st.success("Object Detection Completed")

        # ลบไฟล์ชั่วคราว
        try:
            tfile.close()  # ปิดไฟล์ก่อนลบ
            os.unlink(tfile.name)  # ลบไฟล์
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการลบไฟล์ชั่วคราว: {e}")

# เรียกใช้ฟังก์ชันหลัก
if __name__ == "__main__":
    main()
