import PySimpleGUI as sg
import os, os.path
from screen_capture import ScreenCapture
from PIL import Image, ImageTk
import shutil


screen_capture = ScreenCapture()

sg.theme('Black')
# All the stuff inside your window.
layout = [  [sg.Text('Newest minimap screenshot:')],
            [sg.Image(filename='', key='-IMAGE-')],
            [sg.Combo([str(x) for x in range(22)],default_value='1',key='-number-')],
            [sg.Button('Take new image', key='-snap-'),  sg.Button('Save', key='-save-')] ]

# Create the Window
window = sg.Window('ScavStats', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    if event == '-snap-':
        screen_capture.capture_whole_screen()
        mini_map = screen_capture.crop_minimap_from_current_img()
        mini_map = screen_capture.north_minimap(mini_map, screen_capture.get_degree_from_current_image() * -1)
        mini_map.save("spawnpoints/unsorted/minimap.png", "PNG")

    if event == '-save-':
        nr = str(values["-number-"])
        count = len([name for name in os.listdir('spawnpoints/' + nr + "/") if os.path.isfile(name)])
        shutil.copyfile("spawnpoints/unsorted/minimap.png", "spawnpoints/" + nr + "/" + nr + "-" + str(count) + ".png")
        os.remove("spawnpoints/unsorted/minimap.png")

    if os.path.isfile("spawnpoints/unsorted/minimap.png"):
        # array_img = np.array(Image.open("spawnpoints/unsorted/minimap.png"))
        # imgbytes = cv2.imencode(".png", cv2.cvtColor(array_img, cv2.COLOR_RGB2BGR))[1].tobytes()
        img = Image.open("spawnpoints/unsorted/minimap.png")
        image = ImageTk.PhotoImage(image=img)
        window['-IMAGE-'].update(data=image)
