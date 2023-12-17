# main.py
import tkinter as tk
from tkinter import ttk
from src.data_loader import load_data
from src.cuteness_model import CutenessModel
from torchvision import transforms
from PIL import Image, ImageTk
import torch
import random
import os

class CutenessApp:
    def __init__(self, data_dir, model_path):
        self.data_dir = data_dir
        self.model_path = model_path
        self.model = self.load_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.current_image_paths = None
        self.root = tk.Tk()
        self.root.title("Cuteness App")

        self.create_widgets()

    def load_model(self):
        model = CutenessModel()
        model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def create_widgets(self):
        # Create a frame to hold the images horizontally
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack()

        self.cuteness_scores_label = tk.Label(self.root, text="")
        self.cuteness_scores_label.pack()

        next_button = ttk.Button(self.root, text="Next", command=self.load_new_images)
        next_button.pack()

        quit_button = ttk.Button(self.root, text="Quit", command=self.root.destroy)
        quit_button.pack()

    def display_images(self, images, target_size=(255, 255)):
        # Clear the existing images in the frame
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        # Resize and pack each image label into the horizontal frame
        for image in images:
            image = image.resize(target_size, resample=Image.LANCZOS)
            photo = ImageTk.PhotoImage(image=image)
            label = tk.Label(self.image_frame, image=photo)
            label.image = photo
            label.pack(side="left", padx=10)  # Adjust the padx value for spacing

        self.root.update_idletasks()  # Update the layout

    def load_new_images(self):
        all_image_paths = []

        for root, dirs, files in os.walk(os.path.join(self.data_dir, "train")):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_image_paths.append(os.path.join(root, file))

        if not all_image_paths:
            print("No images found in the 'train' directory.")
            return

        # Get two random image paths from the entire dataset
        self.current_image_paths = random.sample(all_image_paths, 2)

        # Display the images
        images_to_display = [self.load_image_from_path(image_path) for image_path in self.current_image_paths]
        self.display_images(images_to_display)

        # Predict and display cuteness scores
        scores = [self.predict_cuteness_from_path(image_path) for image_path in self.current_image_paths]
        self.cuteness_scores_label.config(text=f"Cuteness Scores: {scores}")

    def load_image_from_path(self, image_path):
        with open(image_path, "rb") as f:
            return Image.open(f).convert("RGB")

    def predict_cuteness_from_path(self, image_path):
        print(f"File path: {image_path}")

        image = self.load_image_from_path(image_path)

        try:
            # Use torchvision.transforms.functional.to_tensor to convert PIL image to tensor
            input_tensor = transforms.functional.to_tensor(image)
            input_tensor = input_tensor.unsqueeze(0)

            # Resize the input tensor to match the expected input size of the model
            input_tensor = torch.nn.functional.interpolate(input_tensor, size=(224, 224), mode='bilinear', align_corners=False)

            print("Input Tensor Shape:", input_tensor.shape)

            with torch.no_grad():
                output = self.model(input_tensor)

            print("Output Tensor:", output)

            return round(output.item(), 2)

        except Exception as e:
            print(f"Error predicting cuteness for file {image_path}: {e}")
            return 0.0

    def run(self):
        self.load_new_images()
        self.root.mainloop()

if __name__ == "__main__":
    data_directory = "data"  # Update with your actual data directory
    model_path = "src/cuteness_model.pth"  # Update with the actual path
    app = CutenessApp(data_directory, model_path)
    app.run()
