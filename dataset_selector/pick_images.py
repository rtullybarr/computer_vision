import csv
import requests
import tkinter
from PIL import Image, ImageTk
import os


def crop_image(species, url_info, filename):

    url = "https://snapshotserengeti.s3.msi.umn.edu/" + url_info
    resp = requests.get(url, stream=True)

    img = Image.open(resp.raw)

    small_img = img.resize((img.size[0] // 2, img.size[1] // 2), Image.ANTIALIAS)

    window = tkinter.Tk()
    window.state('zoomed')
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

        if (click_count % 2) == 0:
            x1 = event.x
            y1 = event.y

            if rectangle is not None:
                canvas.delete(rectangle)

        if (click_count % 2) == 1:
            x2 = event.x
            y2 = event.y

            # draw rectangle
            rectangle = canvas.create_rectangle(x1, y1, x2, y2)

        click_count += 1

    canvas.bind("<Button-1>", callback)

    tkinter.mainloop()

    # image has been closed
    if click_count != 0 and (click_count % 2) == 0:
        image_size = img.size
        # we should have co-ordinates of a box containing the animal
        # fix bounding box issues
        x1 *= 2
        if x1 < 0:
            x1 = 0

        y1 *= 2
        if y1 < 0:
            y1 = 0

        x2 *= 2
        if x2 >= img.size[0]:
            x2 = img.size[0]

        y2 *= 2
        if y2 >= img.size[1]:
            y2 = img.size[1]


        cropped_image = img.crop((x1, y1, x2, y2))
        cropped_image.save(species + "/" + filename)

    # rejected images are not saved, added to list of rejected files
    else:
        with open(species + "/rejected.csv", 'a') as rejected_file:
            rejected_file.write(os.path.basename(url_info) + "\n")


def select_images(species):
    with open(species + '.csv') as csvfile:
        reader = csv.DictReader(csvfile)

        # create directory if it doesn't exist
        os.makedirs(species, exist_ok=True)

        for line in reader:

            # Check if it's a new image
            existing_images = os.listdir(species)
            filename = os.path.basename(line["url_info"])

            if filename in existing_images:
                # skip this one
                continue

            try:
                with open(species + "/rejected.csv", 'r+') as rejected_file:
                    rejected_images = [line.rstrip() for line in rejected_file]

                # Check if it's already been rejected
                if filename in rejected_images:
                    continue

            except FileNotFoundError:
                pass

            crop_image(species, line["url_info"], filename)
            print(filename)


if __name__ == "__main__":
    species_finished = ["wildebeest", "guineaFowl", "giraffe", "impala"]
    species_todo = ["zebra", "hartebeest", "buffalo", "elephant",]

    select_images("hartebeest")

    # alternate mode: provide the url_info and filename
    # e.g.
    # crop_image("wildebeest", "S4/C02/C02_R2/S4_C02_R2_IMAG1205.JPG", "new_filename.JPG")