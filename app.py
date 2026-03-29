from flask import Flask, render_template, Response, jsonify
import cv2
import os
import numpy as np
from datetime import datetime
import csv
import face_recognition

app = Flask(__name__)

video = cv2.VideoCapture(0)

known_encodings = []
known_names = []

# Load known faces
path = "faces"

for file in os.listdir(path):
    img = face_recognition.load_image_file(f"{path}/{file}")
    encodings = face_recognition.face_encodings(img)

    if len(encodings) > 0:
        known_encodings.append(encodings[0])
        known_names.append(os.path.splitext(file)[0])

attendance_marked = False
blink_counter = 0


def markAttendance(name):
    file = "attendance.csv"

    if not os.path.exists(file):
        with open(file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Time"])

    with open(file, "r") as f:
        data = f.readlines()
        names = [line.split(",")[0] for line in data]

    if name not in names:
        with open(file, "a", newline="") as f:
            writer = csv.writer(f)
            current_time = datetime.now().strftime("%H:%M:%S")
            writer.writerow([name, current_time])


def detect_blink(landmarks):
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']

    eye_ratio = (abs(left_eye[1][1] - left_eye[5][1]) +
                 abs(right_eye[1][1] - right_eye[5][1]))

    return eye_ratio < 4


def gen_frames():
    global attendance_marked, blink_counter

    while True:
        success, frame = video.read()

        if not success:
            break

        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        faces = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, faces)
        landmarks = face_recognition.face_landmarks(rgb)

        name = "Unknown"

        for encode, face_loc, landmark in zip(encodings, faces, landmarks):

            matches = face_recognition.compare_faces(known_encodings, encode)
            face_dist = face_recognition.face_distance(known_encodings, encode)

            matchIndex = np.argmin(face_dist)

            if matches[matchIndex]:
                name = known_names[matchIndex]

            if detect_blink(landmark):
                blink_counter += 1

            if blink_counter >= 1 and not attendance_marked and name != "Unknown":
                markAttendance(name)
                attendance_marked = True

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/video')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    global attendance_marked
    if attendance_marked:
        return jsonify({"status": "done"})
    return jsonify({"status": "waiting"})


if __name__ == "__main__":
    app.run(debug=True)