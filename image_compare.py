from image_similarity_measures.quality_metrics import rmse, ssim, sre
import os
import cv2


def calc_closest_val(p_dict, check_max):
    result = {}
    if check_max:
        closest = max(p_dict.values())
    else:
        closest = min(p_dict.values())

    for key, value in p_dict.items():
        if value == closest:
            result[key] = closest

    return result


def collect_all_measures_for_dir(data_dir, original, dim):
    ssim_measures = dict()
    rmse_measures = dict()
    sre_measures = dict()

    for file in os.listdir(data_dir):
        img_path = os.path.join(data_dir, file)
        data_img = cv2.imread(img_path)
        resized_img = cv2.resize(data_img, dim, interpolation=cv2.INTER_AREA)

        ssim_measures[img_path] = ssim(original, resized_img)
        rmse_measures[img_path] = rmse(original, resized_img)
        sre_measures[img_path] = sre(original, resized_img)

    return ssim_measures, rmse_measures, sre_measures


def calc_all_results(ssim_tuple, rmse_tuple, sre_tuple):
    res_ssim = calc_closest_val(ssim_tuple[0], ssim_tuple[1])
    res_rmse = calc_closest_val(rmse_tuple[0], rmse_tuple[1])
    res_sre = calc_closest_val(sre_tuple[0], sre_tuple[1])
    return res_ssim, res_rmse, res_sre


def collect_ssim_measures_for_dir(data_dir, original, dim):
    ssim_measures = dict()

    for file in os.listdir(data_dir):
        img_path = os.path.join(data_dir, file)
        data_img = cv2.imread(img_path)
        resized_img = cv2.resize(data_img, dim, interpolation=cv2.INTER_AREA)

        ssim_measures[img_path] = ssim(original, resized_img)

    return ssim_measures
