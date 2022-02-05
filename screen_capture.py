import os
import sys
import time
import mss
import mss.tools
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageEnhance
import cv2
import numpy as np
import matplotlib.pyplot as plt
import config
import image_similarity_measures
from image_similarity_measures.quality_metrics import rmse, ssim, sre
import pytesseract
import easyocr


class ScreenCapture:

    def __init__(self):
        self.sct = mss.mss()
        self.current_img = None
        self.monitor = None
        self.capturing = True

        # TODO switch to GPU for CUDA support
        self.reader = easyocr.Reader(['en'], gpu=False)  # this needs to run only once to load the model into memory

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
    def crop_minimap_from_current_img(self):
        pil_img = Image.frombytes("RGB", (self.current_img.size.width, self.current_img.size.height),
                                  self.current_img.rgb)

        height, width = pil_img.size
        lum_img = Image.new('L', (height, width), 0)

        draw = ImageDraw.Draw(lum_img)
        # TODO Make these coords relative to the screen-resolution
        draw.pieslice(((57, 96), (275, 318)), 0, 360,
                      fill=255, outline="white")
        img_arr = np.array(pil_img)
        lum_img_arr = np.array(lum_img)

        final_img_arr = np.dstack((img_arr, lum_img_arr))
        return Image.fromarray(final_img_arr[96:318, 57:275])

    def get_degree_from_current_image(self):
        pil_img = Image.frombytes("RGB", (self.current_img.size.width, self.current_img.size.height),
                                  self.current_img.rgb)
        array_img = np.array(pil_img)
        crop_img = Image.fromarray(array_img[62:82, 180:220])
        crop_img = ImageOps.invert(crop_img.convert('L'))
        '''
        enhancer = ImageEnhance.Contrast(crop_img)
        enhancer.enhance(2)
        '''

        # PREPROCESSING 1
        preprocessing1 = np.asarray(crop_img).copy()
        preprocessing1[preprocessing1 < 128] = 0  # Black
        preprocessing1[preprocessing1 >= 128] = 255  # White
        # array2 = cv2.copyMakeBorder(array2, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        preprocessing1 = Image.fromarray(preprocessing1)

        # nr = pytesseract.image_to_string(image=crop_img, config='digits --psm = 9 -c tessedit_write_images=1').rstrip()
        nr = self.reader.readtext(np.array(preprocessing1), detail=0, allowlist="01234567898")
        if not nr:
            # PREPROCESSING 2
            array_img = np.asarray(crop_img).copy()
            blur = cv2.GaussianBlur(array_img, (5, 5), 0)
            ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessing2 = Image.fromarray(th3)
            nr = self.reader.readtext(np.array(preprocessing2), detail=0, allowlist="01234567898")
        print(nr)
        try:
            return int(nr[0].rstrip())
        except ValueError:
            print("Error, NAN")
            crop_img.show()
        except IndexError:
            print("Error, NAN")
            preprocessing1.show()
            preprocessing2.show()

    @staticmethod
    def north_minimap(minimap, degree):
        return minimap.rotate(degree)

    def find_spawn_location(self):
        test_img = cv2.imread('spawnpoints/8/8-2.png')

        ssim_measures = {}
        rmse_measures = {}
        sre_measures = {}

        scale_percent = 100  # percent of original img size
        width = int(test_img.shape[1] * scale_percent / 100)
        height = int(test_img.shape[0] * scale_percent / 100)
        dim = (width, height)

        data_dir = 'spawnpoints/testing_dir'

        for file in os.listdir(data_dir):
            img_path = os.path.join(data_dir, file)
            data_img = cv2.imread(img_path)
            resized_img = cv2.resize(data_img, dim, interpolation=cv2.INTER_AREA)
            ssim_measures[img_path] = ssim(test_img, resized_img)
            rmse_measures[img_path] = rmse(test_img, resized_img)
            sre_measures[img_path] = sre(test_img, resized_img)

        res_ssim = self.calc_closest_val(ssim_measures, True)
        res_rmse = self.calc_closest_val(rmse_measures, False)
        res_sre = self.calc_closest_val(sre_measures, True)

        print("The most similar according to SSIM: ", res_ssim)
        print("The most similar according to RMSE: ", res_rmse)
        print("The most similar according to SRE: ", res_sre)

    @staticmethod
    def calc_closest_val(dict, checkMax):
        result = {}
        if (checkMax):
            closest = max(dict.values())
        else:
            closest = min(dict.values())

        for key, value in dict.items():
            print("The difference between ", key, " and the original image is : \n", value)
            if (value == closest):
                result[key] = closest

        print("The closest value: ", closest)
        print("######################################################################")
        return result

    @staticmethod
    def crop_img():
        path = 'spawnpoints/21/minimap.png'
        img = Image.open(path)
        img = np.array(img)
        img = Image.fromarray(img[96:318, 57:275])
        img.save(path)


if __name__ == "__main__":
    screen_capture = ScreenCapture()
    # print(config.get_config_value_by_id(config.MONITOR_NUMBER))

    screen_capture.capture_whole_screen()
    mini_map = screen_capture.crop_minimap_from_current_img()
    mini_map = screen_capture.north_minimap(mini_map, screen_capture.get_degree_from_current_image() * -1)
    mini_map.save("spawnpoints/unsorted/minimap.png", "PNG")

   # screen_capture.find_spawn_location()

    '''
    while True:
        screen_capture.capture_whole_screen()
        screen_capture.get_degree_from_current_image()
    '''
