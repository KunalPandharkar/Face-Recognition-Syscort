from PIL import Image, ImageTk
import tkinter as tk
import argparse
import datetime
import face_recognition
import cv2
import numpy as np
import os
from tkinter import messagebox

class Application:
    def __init__(self, output_path = "images/"):
        """ Initialize application which uses OpenCV + Tkinter. It displays
            a video stream in a Tkinter window and stores current snapshot on disk """
        self.vs = cv2.VideoCapture(0) # capture video frames, 0 is your default video camera
        self.output_path = output_path  # store output path
        self.current_image = None  # current image from the camera

        self.root = tk.Tk()  # initialize root window
        self.root.title("Python Face Recognition")  # set window title
        # self.destructor function gets fired when the window is closed
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)

        self.panel = tk.Label(self.root)  # initialize image panel
        self.panel.pack(padx=10, pady=10)

        self.name_var=tk.StringVar()
        entry = tk.Entry(self.root, textvariable=self.name_var)
        entry.pack(fill="both", expand=True, padx=10, pady=10)

        # create a button, that when pressed, will take the current frame and save it to file
        self.btn = tk.Button(self.root, text="Snapshot!", command=self.take_snapshot)
        self.btn.pack(fill="both", expand=True, padx=10, pady=10)

        
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

        self.reload_images()

        # start a self.video_loop that constantly pools the video sensor
        # for the most recently read frame
        self.video_loop()

    def video_loop(self):
        """ Get frame from the video stream and show it in Tkinter """
        ret, frame = self.vs.read()  # read frame from video stream
        if ret:  # frame captured without any errors
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            self.rgb_small_frame = small_frame[:, :, ::-1]

            if self.process_this_frame:
                self.face_locations = face_recognition.face_locations(self.rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(self.rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"


                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    try:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                    except Exception as e:
                        pass

                    self.face_names.append(name)

            self.process_this_frame = not self.process_this_frame


            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
            self.current_image = Image.fromarray(cv2image)  # convert image for PIL
            imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
            self.panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
            self.panel.config(image=imgtk)  # show the image
        self.root.after(30, self.video_loop)  # call the same function after 30 milliseconds

    def take_snapshot(self):
        if(self.name_var.get() != ""):
            """ Take snapshot and save it to the file """
            filename = "{}.jpg".format(self.name_var.get())  # construct filename
            p = os.path.join(self.output_path, filename)  # construct output path
            Image.fromarray(self.rgb_small_frame).convert('RGB').save(p, "JPEG")  # save image as jpeg file
            print("[INFO] saved {}".format(filename))
            
            try:
                self.known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file(filename))[0])
                self.known_face_names.append(self.name_var.get())
            except IndexError as e:
                    messagebox.showinfo("Face not detected!")

            self.name_var.set("")
        else:
            messagebox.showwarning("Error", "Enter a valid name")
    
    def reload_images(self):
        self.known_face_encodings = []
        self.known_face_names = []

        for name in os.listdir():
            if (name.endswith(".jpg")) :
                try:
                    self.known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file(name))[0])
                    self.known_face_names.append(name.replace(".jpg", ""))
                except IndexError as e:
                    print(e)
            

    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        self.root.destroy()
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default="./",
    help="path to output directory to store snapshots (default: current folder")
args = vars(ap.parse_args())

# start the app
print("[INFO] starting...")
pba = Application(args["output"])
pba.root.mainloop()