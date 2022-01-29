import time

import mss
import mss.tools
from PIL import Image, ImageDraw
import cv2
import numpy as np
import matplotlib.pyplot as plt
import config


class ScreenCapture:

    def __init__(self):
        self.sct = mss.mss()
        self.current_img = None
        self.monitor = None
        self.capturing = True

        # Read monitor number from config, if -1 get scav_monitor
        if int(config.get_config_value_by_id(config.MONITOR_NUMBER)) == -1:
            config.set_config_by_id(config.MONITOR_NUMBER, self.get_scav_monitor_number())
        self.monitor = self.sct.monitors[int(config.get_config_value_by_id(config.MONITOR_NUMBER))]

    def get_scav_monitor_number(self):
        # Grab screenshots from all monitors and put them together in one image with numbers
        total = list()
        for monitor in self.sct.monitors:
            img_bgr = self.get_monitor_image(monitor)
            prepared_img = self.prepare_monitor_img(img_bgr, monitor)
            total.append(cv2.resize(prepared_img, (500, 350)))

        h_stack = total[0]
        for image in total[1:]:
            h_stack = np.concatenate((h_stack, image), axis=1)

        # TODO Move to actual GUI
        plt.imshow(cv2.cvtColor(h_stack, cv2.COLOR_BGR2RGB))
        plt.show()
        return input("Select the monitor number on which the game will run:")

    def get_monitor_image(self, monitor):
        img = self.sct.grab(monitor)
        conv_img = Image.frombytes("RGB", (img.size.width, img.size.height), img.rgb)
        array_img = np.array(conv_img)
        img_bgr = cv2.cvtColor(array_img, cv2.COLOR_RGB2BGR)
        return img_bgr

    def prepare_monitor_img(self, img, monitor):
        img_result = img
        mon_nr = self.sct.monitors.index(monitor)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_dimensions = (int(img_result.shape[1] / 2.5), int(img_result.shape[0] / 1.5))
        cv2.putText(img_result, str(mon_nr), text_dimensions, font, 20, (0, 0, 255), 40, cv2.LINE_AA)
        img_result = cv2.copyMakeBorder(img_result, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        return img_result

    # Returns mss screenshot element
    def capture_whole_screen(self):
        self.current_img = self.sct.grab(self.monitor)

    # Returns PIL Image
    # TODO Remove debug shows
    def crop_minimap_from_current_img(self):
        pil_img = Image.frombytes("RGB", (self.current_img.size.width, self.current_img.size.height), self.current_img.rgb)
    
        # img = Image.open("main_map.png")
        # plt.imshow(pil_img)
        # plt.show()

        height, width = pil_img.size
        lum_img = Image.new('L', (height, width), 0)

        draw = ImageDraw.Draw(lum_img)
        # TODO Make these coords relative to the screen-resolution
        draw.pieslice(((57, 96), (275, 318)), 0, 360,
                      fill=255, outline="white")
        img_arr = np.array(pil_img)
        lum_img_arr = np.array(lum_img)
        # plt.imshow(Image.fromarray(lum_img_arr))
        # plt.show()
        final_img_arr = np.dstack((img_arr, lum_img_arr))
        # plt.imshow(Image.fromarray(final_img_arr))
        # plt.show()
        # Image.fromarray(final_img_arr).show()
        return Image.fromarray(final_img_arr)



if __name__ == "__main__":
    screen_capture = ScreenCapture()
    # print(config.get_config_value_by_id(config.MONITOR_NUMBER))
    screen_capture.capture_whole_screen()
    screen_capture.crop_minimap_from_current_img()
