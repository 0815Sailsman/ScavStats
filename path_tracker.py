import cv2
import numpy as np
import image_compare

MIN_NR_OF_RESULTS = 50


# TODO Somehow save past-coords, so that they are live-accessible for later, but also nicely stored and organized
class PathTracker:

    # Should be called once you load into the game
    def __init__(self):
        self.MAIN_MAP = cv2.cvtColor(cv2.imread('main_map.png'), cv2.COLOR_BGR2GRAY)

        self.last_coords = (0, 0)

        # Start by instantly getting the spawn location and the fitting coords
        # self.last_coords = self.identify_spawn()

    def match_minimap_and_coords(self, mini_map, last_coords):
        template = cv2.imread(mini_map,0)

        w, h = template.shape[::-1]

        res = cv2.matchTemplate(self.MAIN_MAP, template, cv2.TM_CCOEFF_NORMED)
        threshold = 1.0
        loc = np.where(res >= threshold)
        while len(list(zip(*loc[::-1]))) < MIN_NR_OF_RESULTS:
            threshold -= 0.025
            loc = np.where(res >= threshold)

        best = last_coords
        for pt in zip(*loc[::-1]):
            if abs(best[0] - (pt[0] + int(0.5 * w))) + abs(best[1] - (pt[1] + int(0.5 * h))) < best[0] + best[1]:
                best = (pt[0] + int(0.5 * w), pt[1] + int(0.5 * h))
        return best

    def find_spawn_location(self, minimap):
        minimap = np.asarray(minimap)
        minimap = cv2.cvtColor(minimap, cv2.COLOR_RGB2BGR)

        scale_percent = 100  # percent of original img size
        width = int(minimap.shape[1] * scale_percent / 100)
        height = int(minimap.shape[0] * scale_percent / 100)
        dim = (width, height)

        data_dir = 'spawnpoints/all'

        ssim_measures, rmse_measures, sre_measures = image_compare.collect_all_measures_for_dir(data_dir, minimap, dim)
        res_ssim, res_rmse, res_sre = image_compare.calc_all_results(
            (ssim_measures, True),
            (rmse_measures, False),
            (sre_measures, True))

        ssim_file_name = str(list(res_ssim.keys())[0])
        rmse_file_name = str(list(res_rmse.keys())[0])
        sre_file_name = str(list(res_sre.keys())[0])

        ssim_nr = ssim_file_name.split("\\")[1].split("-")[0]
        rmse_nr = rmse_file_name.split("\\")[1].split("-")[0]
        sre_nr = sre_file_name.split("\\")[1].split("-")[0]

        # TODO Add some logic that every results gets looked at
        return int(ssim_nr)

    def has_current_location(self):
        return self.last_coords != (0, 0)

    # These are in relation to the 4032x4032 map
    def update_coords_from_spawnpoint_nr(self, spawnpoint_nr):
        if spawnpoint_nr == 1:
            self.last_coords = (650, 250)
        elif spawnpoint_nr == 2:
            self.last_coords = (1200, 600)
        elif spawnpoint_nr == 3:
            self.last_coords = (1900, 750)
        elif spawnpoint_nr == 4:
            self.last_coords = (2600, 750)
        elif spawnpoint_nr == 5:
            self.last_coords = (3300, 300)
        elif spawnpoint_nr == 6:
            self.last_coords = (700, 1000)
        elif spawnpoint_nr == 7:
            self.last_coords = (1725, 1375)
        elif spawnpoint_nr == 8:
            self.last_coords = (2500, 1450)
        elif spawnpoint_nr == 9:
            self.last_coords = (3850, 800)
        elif spawnpoint_nr == 10:
            self.last_coords = (225, 2375)
        elif spawnpoint_nr == 11:
            self.last_coords = (1050, 2000)
        elif spawnpoint_nr == 12:
            self.last_coords = (2800, 2275)
        elif spawnpoint_nr == 13:
            self.last_coords = (3450, 1650)
        elif spawnpoint_nr == 14:
            self.last_coords = (1050, 2900)
        elif spawnpoint_nr == 15:
            self.last_coords = (2100, 3050)
        elif spawnpoint_nr == 16:
            self.last_coords = (3275, 2750)
        elif spawnpoint_nr == 17:
            self.last_coords = (3875, 2275)
        elif spawnpoint_nr == 18:
            self.last_coords = (425, 3600)
        elif spawnpoint_nr == 19:
            self.last_coords = (1450, 3750)
        elif spawnpoint_nr == 20:
            self.last_coords = (2800, 3750)
        elif spawnpoint_nr == 21:
            self.last_coords = (3000, 3300)
        elif spawnpoint_nr == 22:
            self.last_coords = (3625, 3625)


if __name__ == "__main__":
    path_tracker = PathTracker()
    print(path_tracker.match_minimap_and_coords('demo_minimap.png', (3600, 1900)))
