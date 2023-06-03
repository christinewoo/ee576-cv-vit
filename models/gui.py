import tkinter as tk
import math
from tkinter import filedialog
from PIL import Image, ImageTk
import vit_pokemon_inf

window = tk.Tk()
window.title('ViT Pokemon Classifer')
window.geometry('800x600')
window.configure(background='white')

def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif")])
    if file_path:
        image, pokemon_name = vit_pokemon_inf(file_path)
        image = Image.open(file_path)
        max_width = 800
        max_height = 600
        width, height = image.size
        aspect_ratio = width / height
        if width > max_width or height > max_height:
            if aspect_ratio > 1:
                width = max_width
                height = int(width / aspect_ratio)
            else:
                height = max_height
                width = int(height * aspect_ratio)

        image = image.resize((width, height))  # Resize the image
        photo = ImageTk.PhotoImage(image)
        result_label.config(image=photo)
        result_label.image = photo
    

image_label = tk.Label(window)
image_label.pack(side=tk.BOTTOM, padx=10, pady=10)

image = Image.open("../gui_pic.jpg")
image = image.resize((300, 100))  # Resize the image to fit the window
photo = ImageTk.PhotoImage(image)
result_label = tk.Label(window)
result_label.pack()
image_label.config(image=photo)
image_label.image = photo

button = tk.Button(window, text="Open Image", command=load_image)

button.pack(padx=10, pady=10)

window.mainloop()