import tkinter as tk

from PIL import Image, ImageTk


class MainWindow:
    def __init__(self):
        self.width = 1280
        self.height = 720

        self.root = tk.Tk()
        self.root.resizable(False, False)
        self.root.geometry(f"{self.width}x{self.height}")

        self.canvas = tk.Canvas(
            self.root,
            highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        # background
        image = Image.open("resources/back.jpg")
        image = image.resize((self.width, self.height))
        self.bg = ImageTk.PhotoImage(image)

        self.canvas.create_image(
            0, 0,
            image=self.bg,
            anchor="nw"
        )

        # text поверх background
        self.text_id = self.canvas.create_text(
            30, 30,
            anchor="nw",
            text="",
            fill="white",
            font=("Consolas", 20)
        )

    def update(self, text):
        self.canvas.itemconfig(
            self.text_id,
            text=text
        )

    def run(self):
        self.root.mainloop()