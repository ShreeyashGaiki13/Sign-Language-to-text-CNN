import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from keras.models import load_model
from filter import filter  # Filter function is imported from a separate file

class SignLanguageApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Real Time Sign Language Conversion to Text")
        self.master.config(bg="#E0E5EC")
        
        # Title Label
        self.title_frame = tk.Frame(master, bg="black")
        self.title_frame.pack(fill="x")
        self.title_label = tk.Label(self.title_frame, text="Sign Language Conversion to Text", font=("Helvetica", 14, "bold"), fg="white", bg="black")
        self.title_label.pack(pady=10)
        
        # Load the trained model
        self.model = load_model("D:\\Mini Project\\Project folder\\sign_language_model.h5")
        
        # Create video frame
        self.video_frame = tk.Label(master, bg="#000000")
        self.video_frame.pack()
        
        # Create output box
        self.output_box = tk.Label(master, bg="#FFFFFF", fg="#000000", font=("Arial", 14), text="Predicted Class: ")
        self.output_box.pack(pady=(20, 0))
        
        # Create exit button
        self.exit_button = tk.Button(master, text="Exit", command=self.exit_app, bg="#FF0000", fg="#FFFFFF", font=("Arial", 14))
        self.exit_button.pack(pady=(10, 20))

        # Start video capture
        self.vid = cv2.VideoCapture(0)
        self.show_frame()

    def show_frame(self):
        # Read frame from the webcam
        _, frame = self.vid.read()

        # Mirror image
        frame = cv2.flip(frame, 1)

        # Resizing the frame window
        frame = cv2.resize(frame, (700, 500))

        # Draw the rectangle on the frame
        cv2.rectangle(frame, (430, 10), (680, 260), (124, 252, 0), 3)

        # Define the region of interest (ROI)
        roi_frame = frame[15:260, 430:670]

        # Apply filter function
        processed_frame = filter(roi_frame)

        # Preprocessing for model
        processed_frame = np.array(processed_frame)
        processed_frame = processed_frame.reshape(1, 48, 48, 1)
        processed_frame = processed_frame/255.0

        # Make predictions using the model
        prediction = self.model.predict(processed_frame)
        predicted_class = np.argmax(prediction)
        
        # Replace class label "10" with "blank"
        if predicted_class == 10:
            predicted_class = "blank"
        
        # Update output box text
        self.output_box.config(text=f"Predicted Class: {predicted_class}")
        
        # Display video frame
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.video_frame.config(image=self.photo)
        self.video_frame.image = self.photo

        # Repeat after 10 milliseconds
        self.master.after(10, self.show_frame)

    def exit_app(self):
        # Ask for confirmation before exiting
        if messagebox.askokcancel("Exit", "Do you want to exit?"):
            self.vid.release()  # Release the webcam
            self.master.destroy()  # Close the application window

def main():
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
