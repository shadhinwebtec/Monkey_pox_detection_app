import tkinter as tk 
from tkinter import filedialog
from tkinter import Label
from PIL import  Image, ImageTk
import numpy as np 
import tensorflow as tf 



#Load the trained model 
model=tf.keras.models.load_model('m_pox_detection.h5')

#preprocess the immage to the format expected by VGG19
def preprocess_image(image_path):
    img=Image.open(image_path).convert('RGB')
    img=img.resize((224,224))
    img=np.array(img)
    img=img/255.0  #Normalize
    img=np.expand_dims(img,axis=0)  #add batch dimension
    return img

# Apply softmax to get probabilities from logits
def predict_mpox(image_path, threshold=0.3):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    
    # Convert prediction (numpy array) to tensor
    prediction_tensor = tf.convert_to_tensor(prediction[0])
    
    # Apply softmax to convert logits to probabilities
    probabilities = tf.nn.softmax(prediction_tensor)
    
    print(f"Raw Prediction (Logits): {prediction}")
    print(f"Probabilities: {probabilities}")
    
    # Get the Mpox probability (class 5)
    mpox_probability = probabilities[5].numpy()
    print(f"Mpox Probability: {mpox_probability}")
    
    # Check if the Mpox probability exceeds the threshold
    if mpox_probability > threshold:
        return "Mpox Detected"
    else:
        return "No Mpox Detected"

# Example usage in the Tkinter UI:
class MpoxApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mpox Detection")
        self.root.geometry("500x500")

        # Title label
        self.label = Label(root, text="Mpox Detection App", font=("Helvetica", 18))
        self.label.pack(pady=20)

        # Accuracy label
        self.accuracy_label = Label(root, text="Model Accuracy: 98.32%", font=("Helvetica", 12), fg="green")
        self.accuracy_label.pack(pady=10)

        # Upload button
        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=10)

        # Result label
        self.result_label = Label(root, text="", font=("Helvetica", 14))
        self.result_label.pack(pady=20)

        # Image display label
        self.image_label = Label(root)
        self.image_label.pack()

    # Function to handle image upload and display
    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path)
            img.thumbnail((300, 300))  # Resize for display
            img = ImageTk.PhotoImage(img)

            # Display image
            self.image_label.config(image=img)
            self.image_label.image = img

            # Get prediction with a threshold of 0.5
            result = predict_mpox(file_path, threshold=0.3)
            self.result_label.config(text=result)

# Main loop for the Tkinter UI
if __name__ == "__main__":
    root = tk.Tk()
    app = MpoxApp(root)
    root.mainloop()       
        
        
          
    