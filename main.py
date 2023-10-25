import csv
import sys
from datetime import datetime

import cv2
import face_recognition
import numpy as np
import os

sys.path.append('/usr/local/lib/python2.7/site-packages')

video_capture = cv2.VideoCapture(0)

jobs_image = face_recognition.load_image_file("jobs.jpg")  # write image address
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]

ratan_tata_image = face_recognition.load_image_file("tata.jpg")  # write image address
ratan_tata_encoding = face_recognition.face_encodings(ratan_tata_image)[0]

sadmona_image = face_recognition.load_image_file("sadmona.jpg")  # write image address
sadmona_encoding = face_recognition.face_encodings(sadmona_image)[0]

tesla_image = face_recognition.load_image_file("tesla.jpg")  # write image address
tesla_encoding = face_recognition.face_encodings(tesla_image)[0]

tesla1_image = face_recognition.load_image_file("tesla1.jpeg")  # write image address
tesla1_encoding = face_recognition.face_encodings(tesla1_image)[0]

known_faces_encoding = [
    jobs_encoding,
    ratan_tata_encoding,
    sadmona_encoding,
    tesla_encoding,
    tesla1_encoding
]

known_faces_names = [
    "Steve Jobs",
    "Ratan Tata",
    "Mona Lisa",
    "Nikola Tesla",
    "Nikola Tesla1"
]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    rgb_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

    if s:
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_names = []
        for face_encodings in face_encodings:
            matches = face_recognition.compare_faces(known_faces_encoding, face_encodings)
            name = ""
            face_distance = face_recognition.face_distance(known_faces_encoding, face_encodings)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])

    cv2.imshow("ADVANCED FACE RECOGNITION SYSTEM USING FR TECHNOLOGY", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
