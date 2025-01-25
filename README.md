**คำแนะนำ

#สร้าง Environment

(Windows) python -m venv venv

(MacOS)python3 -m venv venv

#เข้าถึง Environment

(Windows) .\venv\Scripts\activate

(MacOS) source venv/bin/activate

#ติดตั้ง Library ตามรายการที่ระบุไว้ในไฟล์ txt

pip install -r requirements.txt

#ใช้งานโปรแกรม

streamlit run [program.py]

#เผยแพร่โปรแกรม https://ngrok.com/

ngrok http 8501
