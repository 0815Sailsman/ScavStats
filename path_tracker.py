import cv2 as cv
import numpy as np

MIN_NR_OF_RESULTS = 50


# TODO Somehow access the "video-feed" supplied by main-class or some other object
# TODO Add method that extracts the minimap from the "video-feed"
# TODO Add method that extracts the character rotation from the "video-feed"
# TODO Add method that edits the minimap image so that it is ready for template-matching
# TODO Somehow save past-coords, so that they are live-accessible for later, but also nicely stored and organized
class PathTracker:

    # Should be called once you load into the game
    def __init__(self):
        self.MIN_NR_OF_RESULTS = 50
        self.MAIN_MAP = cv.cvtColor(cv.imread('main_map.png'), cv.COLOR_BGR2GRAY)

        self.last_coords = (0, 0)

        # Start by instantly getting the spawn location and the fitting coords
        # self.last_coords = self.identify_spawn()

    def match_minimap_and_coords(self, mini_map, last_coords):
        template = cv.imread(mini_map,0)

        w, h = template.shape[::-1]

        res = cv.matchTemplate(self.MAIN_MAP, template, cv.TM_CCOEFF_NORMED)
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


if __name__ == "__main__":
    path_tracker = PathTracker()
    print(path_tracker.match_minimap_and_coords('demo_minimap.png', (3600, 1900)))
