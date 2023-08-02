import tensorflow_hub as hub
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageTk

model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

def stylize_image(content_image, style_image):
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
    return stylized_image

def save_image(stylized_image, output_path):
    cv2.imwrite(output_path, cv2.cvtColor(np.squeeze(stylized_image)*255, cv2.COLOR_BGR2RGB))

def show_image(image_path):
    img = Image.open(image_path)
    #img = img.resize((300, 300))  # Resize the image to fit in the window
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk  # Keep a reference to avoid garbage collection

def download_image():
    save_path = filedialog.asksaveasfilename(defaultextension='.jpg', filetypes=[('JPEG Image', '*.jpg')])
    if save_path:
        save_image(stylized_image, save_path)
        print('Image saved to {}'.format(save_path))

root = tk.Tk()
root.title('Image Stylization')

content_image_label = tk.Label(root, text='Content Image')
content_image_label.grid(row=0, column=0)

content_image_path = filedialog.askopenfilename(title='Select Content Image')

style_image_label = tk.Label(root, text='Style Image')
style_image_label.grid(row=1, column=0)

style_image_path = filedialog.askopenfilename(title='Select Style Image')

content_image = load_image(content_image_path)
style_image = load_image(style_image_path)

stylized_image = stylize_image(content_image, style_image)

output_path = 'generated_img.jpg'
save_image(stylized_image, output_path)

# Show the generated image
image_label = tk.Label(root)
image_label.grid(row=2, column=0, columnspan=2)
show_image(output_path)

# Add a button to download the image
download_button = tk.Button(root, text='Download Image', command=download_image)
download_button.grid(row=3, column=0, columnspan=2)

root.mainloop()
