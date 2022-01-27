import cv2 as cv
import numpy as np

MIN_NR_OF_RESULTS = 50


# Somehow pipe in images and last coords
def match_minimap_and_coords(mini_map, last_coords):
    img_rgb = cv.imread('main_map.png')
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    template = cv.imread(mini_map,0)

    w, h = template.shape[::-1]

    res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
    threshold = 1.0
    loc = np.where(res >= threshold)
    while len(list(zip(*loc[::-1]))) < MIN_NR_OF_RESULTS:
        threshold -= 0.025
        loc = np.where(res >= threshold)

    best = last_coords
    for pt in zip(*loc[::-1]):
        if abs(best[0] - (pt[0] + int(0.5 * w))) + abs(best[1] - (pt[1] + int(0.5 * h))) < best[0] + best[1]:
            best = (pt[0] + int(0.5 * w), pt[1] + int(0.5 * h))
        # print(pt[0] + int(0.5 * w))
        # print(pt[1] + int(0.5 * h))
    return best


if __name__ == "__main__":
    print(match_minimap_and_coords('demo_minimap.png', (3600, 1900)))
