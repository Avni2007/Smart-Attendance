import cv2
import csv
import time
import os
import numpy as np
from datetime import datetime
from flask import Flask, render_template, Response
import mediapipe as mp
import face_recognition

# ==============================
# Flask App
# ==============================
app = Flask(__name__)

# ==============================
# LOAD KNOWN FACES
# ==============================
known_encodings = []
known_names = []

folder = "known_faces"

if not os.path.exists(folder):
    os.makedirs(folder)

for file in os.listdir(folder):

    if file.endswith(".jpg") or file.endswith(".png"):

        path = os.path.join(folder, file)

        try:
            image = face_recognition.load_image_file(path)

            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:

                known_encodings.append(encodings[0])

                known_names.append(
                    os.path.splitext(file)[0]
                )

        except Exception as e:
            print("Face Load Error:", e)

print("✅ Loaded Students:", known_names)

# ==============================
# MEDIAPIPE SETUP
# ==============================
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=3,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==============================
# CAMERA SETUP
# ==============================
def initialize_camera():

    for index in range(3):

        try:

            cap = cv2.VideoCapture(
                index,
                cv2.CAP_DSHOW
            )

            if cap.isOpened():

                cap.set(
                    cv2.CAP_PROP_FRAME_WIDTH,
                    640
                )

                cap.set(
                    cv2.CAP_PROP_FRAME_HEIGHT,
                    480
                )

                cap.set(
                    cv2.CAP_PROP_FPS,
                    30
                )

                ret, frame = cap.read()

                if ret:

                    print(
                        f"✅ Camera Opened at Index {index}"
                    )

                    return cap

                else:

                    cap.release()

        except Exception as e:

            print("Camera Error:", e)

    return None


cap = initialize_camera()

time.sleep(2)

if cap is None:
    print("❌ Camera Not Accessible")

# ==============================
# CSV FILE
# ==============================
attendance_file = "attendance.csv"

# ==============================
# GLOBAL VARIABLES
# ==============================
marked_students = set()

popup_name = None

eye_status = {}

frame_count = 0

# ==============================
# EYE LANDMARKS
# ==============================
LEFT_EYE = [33, 160, 158, 133, 153, 144]

RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ==============================
# EAR FUNCTION
# ==============================
def eye_aspect_ratio(
    landmarks,
    eye_indices,
    w,
    h
):

    points = [
        (
            int(landmarks[i].x * w),
            int(landmarks[i].y * h)
        )
        for i in eye_indices
    ]

    v1 = np.linalg.norm(
        np.array(points[1]) - np.array(points[5])
    )

    v2 = np.linalg.norm(
        np.array(points[2]) - np.array(points[4])
    )

    h1 = np.linalg.norm(
        np.array(points[0]) - np.array(points[3])
    )

    ear = (v1 + v2) / (2.0 * h1)

    return ear

# ==============================
# ATTENDANCE FUNCTION
# ==============================
def mark_attendance(name):

    if name in marked_students:
        return

    now = datetime.now()

    dt = now.strftime(
        '%Y-%m-%d %H:%M:%S'
    )

    file_exists = os.path.isfile(
        attendance_file
    )

    with open(
        attendance_file,
        'a',
        newline=''
    ) as f:

        writer = csv.writer(f)

        if not file_exists:

            writer.writerow([
                "Name",
                "Time",
                "Status"
            ])

        writer.writerow([
            name,
            dt,
            "Present"
        ])

    marked_students.add(name)

    print(f"✅ {name} Marked Present")

# ==============================
# FRAME GENERATOR
# ==============================
def gen_frames():

    global popup_name
    global frame_count
    global cap

    while True:

        # Reconnect Camera
        if cap is None or not cap.isOpened():

            print("🔄 Reconnecting Camera")

            cap = initialize_camera()

            time.sleep(1)

            continue

        success, frame = cap.read()

        # Camera Fail Protection
        if not success or frame is None:

            print("❌ Failed To Grab Frame")

            cap.release()

            cap = initialize_camera()

            time.sleep(1)

            continue

        # Flip Frame
        frame = cv2.flip(frame, 1)

        frame_count += 1

        # Skip Alternate Frames
        if frame_count % 2 != 0:

            ret, buffer = cv2.imencode(
                '.jpg',
                frame
            )

            frame_bytes = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                frame_bytes +
                b'\r\n'
            )

            continue

        # Resize Frame
        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=0.5,
            fy=0.5
        )

        rgb_small = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )

        # Face Recognition
        face_locations = (
            face_recognition.face_locations(
                rgb_small
            )
        )

        face_encodings = (
            face_recognition.face_encodings(
                rgb_small,
                face_locations
            )
        )

        # Scale Back
        face_locations = [
            (t*2, r*2, b*2, l*2)
            for (t, r, b, l)
            in face_locations
        ]

        results = None

        if len(face_locations) > 0:

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = face_mesh.process(rgb)

        if results and results.multi_face_landmarks:

            for i, (
                (top, right, bottom, left),
                face_landmarks
            ) in enumerate(
                zip(
                    face_locations,
                    results.multi_face_landmarks
                )
            ):

                name = "Unknown"

                # Match Face
                if i < len(face_encodings):

                    face_encoding = (
                        face_encodings[i]
                    )

                    matches = (
                        face_recognition.compare_faces(
                            known_encodings,
                            face_encoding
                        )
                    )

                    face_distances = (
                        face_recognition.face_distance(
                            known_encodings,
                            face_encoding
                        )
                    )

                    if len(face_distances) > 0:

                        best_match_index = (
                            np.argmin(
                                face_distances
                            )
                        )

                        if matches[
                            best_match_index
                        ]:

                            name = known_names[
                                best_match_index
                            ]

                # Draw Rectangle
                cv2.rectangle(
                    frame,
                    (left, top),
                    (right, bottom),
                    (0, 255, 0),
                    2
                )

                # Name Text
                cv2.putText(
                    frame,
                    name,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                # Blink Detection
                if name != "Unknown":

                    h, w, _ = frame.shape

                    left_ear = eye_aspect_ratio(
                        face_landmarks.landmark,
                        LEFT_EYE,
                        w,
                        h
                    )

                    right_ear = eye_aspect_ratio(
                        face_landmarks.landmark,
                        RIGHT_EYE,
                        w,
                        h
                    )

                    ear = (
                        left_ear +
                        right_ear
                    ) / 2.0

                    if name not in eye_status:

                        eye_status[name] = True

                    # Eyes Closed
                    if ear < 0.20:

                        eye_status[name] = False

                    # Blink Detected
                    else:

                        if eye_status[name] == False:

                            eye_status[name] = True

                            if name not in marked_students:

                                mark_attendance(name)

                                popup_name = name

        # Encode Frame
        ret, buffer = cv2.imencode(
            '.jpg',
            frame
        )

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )

        time.sleep(0.03)

# ==============================
# ROUTES
# ==============================
@app.route('/')
def index():

    return render_template(
        'index.html'
    )

@app.route('/video_feed')
def video_feed():

    return Response(
        gen_frames(),
        mimetype=(
            'multipart/x-mixed-replace; '
            'boundary=frame'
        )
    )

@app.route('/check_popup')
def check_popup():

    global popup_name

    if popup_name:

        name = popup_name

        popup_name = None

        return name

    return "none"

# ==============================
# RUN APP
# ==============================
if __name__ == "__main__":

    try:

        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )

    finally:

        if cap:
            cap.release()

        cv2.destroyAllWindows()