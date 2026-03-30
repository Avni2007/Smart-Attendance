# Smart Attendance System with Liveness Detection

## 📘 Overview

The **Smart Attendance System with Liveness Detection** is a secure attendance management solution that automates attendance marking using **face recognition** and prevents proxy attendance through **blink-based liveness detection**.

This system ensures that attendance is recorded only when a **registered user is detected** and **live presence is verified**, making attendance secure, fast, and contactless.

---

## ✨ Features

* Real-time face detection using webcam
* Face recognition for registered users
* Blink detection for liveness verification
* Automatic attendance marking
* Attendance record storage in CSV / database
* Web-based attendance dashboard

---

## 🛠 Technologies Used

* **Python**
* **OpenCV**
* **MediaPipe**
* **Flask**
* **HTML, CSS, JavaScript**
* **CSV / SQLite**

---

## ⚙️ How It Works

1. User appears before webcam
2. Face is detected
3. Face is matched with registered data
4. Blink detection confirms live presence
5. Attendance is marked automatically
6. Record is stored digitally

---

## 📂 Project Structure

```text
Smart-Attend/
│── faces/
│── static/
│── templates/
│── attendance.csv
│── app.py
│── requirements.txt
│── README.md
```

---

## 🚀 Installation

```bash
git clone https://github.com/Avni2007/Smart-Attend.git
cd Smart-Attend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

---

## 🎯 Applications

* Schools and Colleges
* Offices
* Laboratories
* Examination Halls

---

## 🔮 Future Scope

* Cloud-based attendance storage
* Mobile app integration
* Voice verification
* Multi-camera support

---

## 📌 Conclusion

This project improves attendance accuracy and security by combining face recognition with liveness detection, providing a practical and reliable modern attendance solution.

---



