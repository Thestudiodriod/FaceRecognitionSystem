import face_recognition
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime


from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText




db_path = '/usr/bin'
db_file = 'Attendance_sys.db'
db_full_path = os.path.join(db_path, db_file)
predictor_path = "shape_predictor_68_face_landmarks.dat"

# engine = create_engine('sqlite:///' + db_full_path)

Base = declarative_base()



def send_email(recipient_email, student_name):
    sender_email = "your_email@gmail.com" 
    sender_password = "your_password"  

    # Create the email message
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = recipient_email
    message['Subject'] = 'Attendance Confirmation'
    body = f"Hello {student_name}, your attendance  has been taken for today at  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
    message.attach(MIMEText(body, 'plain'))

    # Connect to Gmail SMTP server
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = message.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        print(f"Email sent to {student_name} at {recipient_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

class Attendance(Base):
    __tablename__ = 'attendance'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    status = Column(String, default="Present")
    time = Column(String)
    date = Column(String)

engine = create_engine('sqlite:///db_files/sys.db') #('sqlite:///{}'.format(db_full_path))
Base.metadata.create_all(engine)


Session = sessionmaker(bind=engine)
session = Session()
#function to take the attendance in a csv file
def attendance_csv(name):
    with open('db_files/attendance.csv', 'r+') as f:
        MyDataList = f.readlines()  # Use readlines() instead of readline()
        nameList = []
        for line in MyDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            is_there = "Present"
            f.writelines(f'\n{name},{is_there},{tStr},{dStr}')
            

def attendance(name):
    # Check if the name already exists in the database
    existing_entry = session.query(Attendance).filter_by(name=name).first()
    if not existing_entry:
        time_now = datetime.now()
        new_attendance = Attendance(
            name=name,
            time=time_now.strftime('%H:%M:%S'),
            date=time_now.strftime('%d/%m/%Y')
        )
        session.add(new_attendance)
        session.commit()
        # student_emails ={'name': "student@school.com"}
        # student_email = student_emails[name]  # You need to define `student_emails` dictionary
        # send_email(student_email, name)


from deepface import DeepFace
def detect_emotion(face_image):
    try:
        analysis = DeepFace.analyze(face_image, actions=['emotion'])
        return analysis['dominant_emotion']
    except Exception as e:
        print(f"Emotion detection failed: {e}")
        return None

from sqlalchemy.orm import sessionmaker

Session = sessionmaker(bind=engine)

def add_attendance(name):
    session = Session()  # Create a session instance
    try:
        # Query the existing entry
        existing_entry = session.query(Attendance).filter_by(name=name).first()
        
        # If the entry does not exist, add a new record
        if not existing_entry:
            time_now = datetime.now()
            new_attendance = Attendance(
                name=name,
                time=time_now.strftime('%H:%M:%S'),
                date=time_now.strftime('%d/%m/%Y')
            )
            session.add(new_attendance)
            session.commit()  # Commit the transaction manually
    except Exception as e:
        session.rollback()  # Roll back in case of exception
        print(f"An error occurred: {e}")
    finally:
        session.close()  # Always close the session to free resources


#code starts here calling the openCV video function
video_capture = cv2.VideoCapture(0)

#training model to get encoding for faces

known_face_encodings = []
known_face_names = []

path = 'sample_images'
path_list = []
for a in os.listdir(path)[1:]:
    myList = os.path.join(path, a)
    path_list.append(myList)
    #studentName.append
    known_face_names.append(a.split(".", 1)[0])

for i in path_list:
    training_img = face_recognition.load_image_file(i)
    # print(face_recognition.face_encodings(training_img, model="large"))
    training_img_encoding = face_recognition.face_encodings(training_img, model="large")[0]
    
    known_face_encodings.append(training_img_encoding)
    
    
    
#initializing variables to take face location
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

#opening system camera
ret, frame = video_capture.read()


TP = 0  # True Positives
FP = 0  # False Positives
FN = 0  # False Negatives

unknown_faces_dir = 'unknown_faces'
if not os.path.exists(unknown_faces_dir):
    os.makedirs(unknown_faces_dir)

def save_unknown_face(frame, top, right, bottom, left):
    face_image = frame[top:bottom, left:right]  # Crop the face from the frame
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    face_filename = f"{unknown_faces_dir}/unknown_{timestamp}.jpg"
    cv2.imwrite(face_filename, face_image)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, )
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                face_names.append(name)
                TP += 1  # Recognized correctly
            else:
                FN += 1  # Not recognized correctly
                face_names.append(name)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4


        #remove the comment to use deepface 
        face_frame = frame[top:bottom, left:right]
        emotion = detect_emotion(face_frame)  # Detect emotion

        if name == "Unknown":
            save_unknown_face(frame, top, right, bottom, left)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        attendance_csv(name)
        add_attendance(name)
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




if TP + FP > 0 and TP + FN > 0:
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f"F1 Score: {f1_score}")
    else:
        f1_score = 0
        print("No faces processed to calculate F1 score.")
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
print('total poll is: ' + str(len(name)))
