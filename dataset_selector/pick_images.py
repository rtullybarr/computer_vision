import csv
import requests
import tkinter
from PIL import Image, ImageTk
import os


def select_images(species):
    with open(species + '.csv') as csvfile:
        reader = csv.DictReader(csvfile)

        # create directory if it doesn't exist
        os.makedirs(species, exist_ok=True)

        for line in reader:
            url = "https://snapshotserengeti.s3.msi.umn.edu/" + line["url_info"]
            resp = requests.get("https://snapshotserengeti.s3.msi.umn.edu/" + line["url_info"], stream=True)

            img = Image.open(resp.raw)
            small_img = img.resize((img.size[0]//2, img.size[1]//2), Image.ANTIALIAS)

            window = tkinter.Tk()
            canvas = tkinter.Canvas(window, width=small_img.size[0], height=small_img.size[1])
            canvas.pack()
            img_tk = ImageTk.PhotoImage(small_img)
            canvas.create_image(0, 0, image=img_tk, anchor="nw")

            x1 = 0
            y1 = 0
            x2 = 0
            y2 = 0
            click_count = 0
            rectangle = None

            # crop image
            def callback(event):
                nonlocal click_count
                nonlocal x1
                nonlocal y1
                nonlocal x2
                nonlocal y2

                nonlocal rectangle

                if (click_count%2) == 0:
                    x1 = event.x
                    y1 = event.y

                    if rectangle is not None:
                        canvas.delete(rectangle)

                if (click_count%2) == 1:
                    x2 = event.x
                    y2 = event.y

                    # draw rectangle
                    rectangle = canvas.create_rectangle(x1, y1, x2, y2)

                click_count += 1

            canvas.bind("<Button-1>", callback)

            tkinter.mainloop()

            # image has been closed
            if click_count != 0 and (click_count%2) == 0:
                # we should have co-ordinates of a box containing the animal
                cropped_img = img.crop((x1*2, y1*2, x2*2, y2*2))
                filename = os.path.basename(line["url_info"])
                cropped_img.save(species + "/" + filename)

            # rejected images are not saved


if __name__ == "__main__":
    species_finished = ["wildebeest"]
    species_todo = ["zebra", "hartebeest", "buffalo", "impala", "giraffe", "elephant", "guineaFowl"]
    select_images("zebra")
