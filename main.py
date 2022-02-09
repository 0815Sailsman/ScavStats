import time

import PySimpleGUI as sg
from screen_capture import ScreenCapture
from path_tracker import PathTracker
from PIL import Image, ImageTk, ImageDraw


screen_capture = ScreenCapture()
path_tracker = PathTracker()

sg.theme('Black')
# All the stuff inside your window.
layout = [
    [sg.Push(), sg.Text("Path travelled ", font=("Arial", 20), pad=(200, 20)), sg.Push(), sg.Text("CURRENT STATE", font=("Arial", 20), key="-CATEGORY-"), sg.Push()],
    [sg.Push(), sg.Image(filename='small_map.png', key='-MAP-', size=(512, 512)), sg.Push(), sg.Push(),sg.Push()],
    [sg.VPush()]
]

# Create the Window
window = sg.Window('ScavStats', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:                    #
    event, values = window.read(timeout=1)
    if event == sg.WIN_CLOSED:
        break
    screen_capture.capture_whole_screen()

    # Ingame, loading screen, map, crafting menu, scoreboard...
    screen_capture.categorize()
    window["-CATEGORY-"].update(screen_capture.current_category)

    # Only when actually ingame
    if screen_capture.current_category == "ingame":
        # Check if we already have located the spawn
        if not path_tracker.has_current_location():
            # Further wait for fade in
            time.sleep(1)
            mini_map = screen_capture.crop_minimap_from_current_img()
            mini_map = screen_capture.north_minimap(mini_map, screen_capture.get_degree_from_current_image() * -1)
            spawnpoint_nr = path_tracker.find_spawn_location(mini_map)
            path_tracker.update_coords_from_spawnpoint_nr(spawnpoint_nr)
            img = Image.open("main_map.png").convert('RGB')
            # TODO Extract draw method
            draw = ImageDraw.Draw(img)
            x = path_tracker.last_coords[0]
            y = path_tracker.last_coords[1]
            draw.ellipse((x-50, y-50, x+50, y+50), fill = 'red', outline ='red')

            # TODO Extract img_update method
            img = img.resize((512, 512))
            window['-MAP-'].update(data=ImageTk.PhotoImage(img))
