import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import requests

API_URL = "http://127.0.0.1:8000/predict"


def upload_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = Image.open(file_path)
    img.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img)

    panel.config(image=img_tk)
    panel.image = img_tk

    with open(file_path, "rb") as f:
        response = requests.post(API_URL, files={"file": f})

    data = response.json()
    result_label.config(
        text=f"Emotion: {data['emotion']} ({data['confidence'] * 100:.2f}%)"
    )


root = tk.Tk()
root.title("Emotion Recognition")

btn = tk.Button(root, text="Upload Image", command=upload_image)
btn.pack()

panel = tk.Label(root)
panel.pack()

result_label = tk.Label(root, font=("Arial", 16))
result_label.pack()

root.mainloop()
