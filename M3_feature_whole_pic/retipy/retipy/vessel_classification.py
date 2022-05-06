import numpy as np
import cv2
import h5py
import glob
import os
from keras.models import model_from_json
from retipy import retina
from retipy import landmarks as l

"""Module with operations related to classify vessels into arteries and veins."""

_base_directory_training = 'retipy/resources/images/drive/training/'
_base_directory_test = 'retipy/resources/images/drive/test/'
_base_directory_model = os.path.join(os.path.dirname(__file__), 'model/')


def _vessel_widths(center_img: np.ndarray, segmented_img: np.ndarray):
    image = segmented_img.copy()
    widths = []
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if center_img[i, j] == 255:
                w0 = 0
                w45 = 0
                w90 = 0
                w135 = 0
                w180 = 1
                w225 = 1
                w270 = 1
                w315 = 1
                while True:
                    if image[i, j + w0 + 1] != 0:
                        w0 += 1
                    if image[i, j - w180 - 1] != 0:
                        w180 += 1
                    if image[i - w90 - 1, j] != 0:
                        w90 += 1
                    if image[i + w270 + 1, j] != 0:
                        w270 += 1
                    if image[i - w45 - 1, j + w45 + 1] != 0:
                        w45 += 1
                    if image[i + w225 + 1, j - w225 - 1] != 0:
                        w225 += 1
                    if image[i - w135 - 1, j - w135 - 1] != 0:
                        w135 += 1
                    if image[i + w315 + 1, j + w315 + 1] != 0:
                        w315 += 1

                    if image[i, j + w0 + 1] == 0 and image[i, j - w180 - 1] == 0:
                        widths.append([i, j, 0, w0, w180])
                        break
                    elif image[i - w90 - 1, j] == 0 and image[i + w270 + 1, j] == 0:
                        widths.append([i, j, 90, w90, w270])
                        break
                    elif image[i - w45 - 1, j + w45 + 1] == 0 and image[i + w225 + 1, j - w225 - 1] == 0:
                        widths.append([i, j, 45, w45, w225])
                        break
                    elif image[i - w135 - 1, j - w135 - 1] == 0 and image[i + w315 + 1, j + w315 + 1] == 0:
                        widths.append([i, j, 135, w135, w315])
                        break
    return widths


def _local_binary_pattern(window: list):
    x = [0, 0, 1, 2, 2, 2, 1, 0]
    y = [1, 2, 2, 2, 1, 0, 0, 0]
    decimal = 0
    center = window[1][1]
    for i in range(0, 8):
        if center >= window[x[i]][y[i]]:
            decimal += np.power(2, i)
    return decimal


def _preparing_data(widths: list, sections: int, original_img: np.ndarray, classified_img: np.ndarray,
                   bright_img: np.ndarray, gray_img: np.ndarray):
    f_vectors = []
    if classified_img is not None:
        for w in widths:
            w0 = w[0]
            w1 = w[1]
            if (np.array_equal(classified_img[w0, w1], [255, 0, 0]) or np.array_equal(classified_img[w0, w1], [0, 0, 255])) \
                    and ((w[3]+w[4]) >= 2):
                if np.array_equal(classified_img[w0, w1], [255, 0, 0]):
                    out = 0
                elif np.array_equal(classified_img[w0, w1], [0, 0, 255]):
                    out = 1

                iv = _vector(w, sections, original_img, bright_img, gray_img, out)
                f_vectors.append(iv)
    else:
        for w in widths:
            if (w[3] + w[4]) >= 2:
                iv = _vector(w, sections, original_img, bright_img, gray_img, -1)
                f_vectors.append(iv)

    return f_vectors


def _vector(w: list, sections: int, original_img: np.ndarray, bright_img: np.ndarray, gray_img: np.ndarray, out: int):
    iv = []
    w0 = w[0]
    w1 = w[1]
    angle = w[2]
    section = (w[3] + w[4]) / sections
    if angle == 0:
        y = w1 + w[3]
        iv.extend([w0, w1])
        for i in range(0, sections + 1):
            step = int(np.floor(y - (i * section)))
            iv.extend(original_img[w0, step])
            iv.extend([bright_img[w0, step]])
            iv.extend([_local_binary_pattern(gray_img[w0 - 1:w0 + 2, step - 1:step + 2])])
        iv.extend([w[3] + w[4]])
        iv.extend([out])
    elif angle == 45:
        x = w0 - w[3]
        y = w1 + w[3]
        iv.extend([w0, w1])
        for i in range(0, sections + 1):
            stepx = int(np.floor(x + (i * section)))
            stepy = int(np.floor(y - (i * section)))
            iv.extend(original_img[stepx, stepy])
            iv.extend([bright_img[stepx, stepy]])
            iv.extend([_local_binary_pattern(gray_img[stepx - 1:stepx + 2, stepy - 1:stepy + 2])])
        iv.extend([w[3] + w[4]])
        iv.extend([out])
    elif angle == 90:
        x = w0 - w[3]
        iv.extend([w0, w1])
        for i in range(0, sections + 1):
            step = int(np.floor(x + (i * section)))
            iv.extend(original_img[step, w1])
            iv.extend([bright_img[step, w1]])
            iv.extend([_local_binary_pattern(gray_img[step - 1:step + 2, w1 - 1:w1 + 2])])
        iv.extend([w[3] + w[4]])
        iv.extend([out])
    elif angle == 135:
        x = w0 - w[3]
        y = w1 - w[3]
        iv.extend([w0, w1])
        for i in range(0, sections + 1):
            stepx = int(np.floor(x + (i * section)))
            stepy = int(np.floor(y + (i * section)))
            iv.extend(original_img[stepx, stepy])
            iv.extend([bright_img[stepx, stepy]])
            iv.extend([_local_binary_pattern(gray_img[stepx - 1:stepx + 2, stepy - 1:stepy + 2])])
        iv.extend([w[3] + w[4]])
        iv.extend([out])

    return iv


def _feature_vectors():
    directory = _base_directory_training + "original/"
    features = []
    for filename in sorted(glob.glob(os.path.join(directory, '*.tif'))):
        name = os.path.basename(filename)
        name = name.split(".")[0]

        original = cv2.imread(filename, 1)
        gray = cv2.imread(filename, 0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

        lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)

        manual = retina.Retina(None, _base_directory_training + "manual/" + name + ".png")
        manual.threshold_image()
        thr_img = manual.get_uint_image()
        cv2.circle(thr_img, maxLoc, 60, 0, -1)
        manual.skeletonization()
        skeleton_img = manual.get_uint_image()
        cv2.circle(skeleton_img, maxLoc, 60, 0, -1)
        landmarks, segmented_skeleton_img = l.potential_landmarks(skeleton_img, 3)

        av = cv2.imread(_base_directory_training + "av/" + name + ".png", 1)

        widths = _vessel_widths(segmented_skeleton_img, thr_img)
        data = _preparing_data(widths, 6, original, av, L, gray)
        features.extend(data)

    h5f = h5py.File(_base_directory_model + 'vector_features_interpolation.h5', 'w')
    h5f.create_dataset('training', data=features)
    return features


def _loading_model(original: np.ndarray, threshold: np.ndarray, av: np.ndarray, size: int):
    # Load model of the neuronal network
    json_file = open(_base_directory_model + 'modelVA.json', "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # Load weights
    loaded_model.load_weights(_base_directory_model + 'modelVA.h5')

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    manual = retina.Retina(threshold, None)
    manual.threshold_image()
    thr_img = manual.get_uint_image()
    cv2.circle(thr_img, maxLoc, 60, 0, -1)
    manual.skeletonization()
    skeleton_img = manual.get_uint_image()
    cv2.circle(skeleton_img, maxLoc, 60, 0, -1)
    landmarks, segmented_skeleton_img = l.potential_landmarks(skeleton_img, 3)

    widths = _vessel_widths(segmented_skeleton_img, thr_img)
    data = _preparing_data(widths, 6, original, av, L, gray)

    features = np.array(data)
    predict_img = np.full((segmented_skeleton_img.shape[0], segmented_skeleton_img.shape[1]), 3, dtype=float)

    for row in range(0, features.shape[0]):
        prediction = loaded_model.predict(np.divide(features[row:row + 1, 2:size], 255), batch_size=1)
        predict_img[features[row, 0], features[row, 1]] = prediction

    return features, segmented_skeleton_img, thr_img, predict_img


def _validating_model(features: np.ndarray, skeleton_img: np.ndarray, original_img: np.ndarray, predicted_img: np.ndarray, size: int, av: int):
    max_acc = -1
    rgb_prediction = []
    network_prediction = []
    original = []
    if av == 0:
        manual_copy = retina.Retina(skeleton_img, None)
        manual_copy.bin_to_bgr()
        manual_copy = manual_copy.get_uint_image()
        original_copy = original_img.copy()
        predict_copy = predicted_img.copy()
        mask0 = predict_copy == 3
        mask1 = (predict_copy >= 0) & (predict_copy < 0.8)
        mask2 = (predict_copy >= 0.8) & (predict_copy <= 1)
        predict_copy[mask1] = 1
        predict_copy[mask2] = 2
        predict_copy[mask0] = 0
        for row in range(0, features.shape[0]):
            if predict_copy[features[row, 0], features[row, 1]] == 2:
                manual_copy[features[row, 0], features[row, 1]] = [0, 0, 255]
                original_copy[features[row, 0], features[row, 1]] = [0, 0, 255]
            elif predict_copy[features[row, 0], features[row, 1]] == 1:
                manual_copy[features[row, 0], features[row, 1]] = [255, 0, 0]
                original_copy[features[row, 0], features[row, 1]] = [255, 0, 0]

        rgb_prediction = manual_copy
        network_prediction = predict_copy
        original = original_copy
    else:
        for i in range(0, 1000):
            manual_copy = retina.Retina(skeleton_img, None)
            manual_copy.bin_to_bgr()
            manual_copy = manual_copy.get_uint_image()
            original_copy = original_img.copy()
            predict_copy = predicted_img.copy()
            k = 0.001*i
            mask0 = predict_copy == 3
            mask1 = (predict_copy >= 0) & (predict_copy < k)
            mask2 = (predict_copy >= k) & (predict_copy <= 1)
            predict_copy[mask1] = 1
            predict_copy[mask2] = 2
            predict_copy[mask0] = 0
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0
            for row in range(0, features.shape[0]):
                if predict_copy[features[row, 0], features[row, 1]] == 2:
                    manual_copy[features[row, 0], features[row, 1]] = [0, 0, 255]
                    original_copy[features[row, 0], features[row, 1]] = [0, 0, 255]
                elif predict_copy[features[row, 0], features[row, 1]] == 1:
                    manual_copy[features[row, 0], features[row, 1]] = [255, 0, 0]
                    original_copy[features[row, 0], features[row, 1]] = [255, 0, 0]

                if int(predict_copy[features[row, 0], features[row, 1]]) == 1 and features[row, size] == 0:
                    true_negative += 1
                elif int(predict_copy[features[row, 0], features[row, 1]]) == 2 and features[row, size] == 1:
                    true_positive += 1
                elif int(predict_copy[features[row, 0], features[row, 1]]) == 2 and features[row, size] == 0:
                    false_positive += 1
                elif int(predict_copy[features[row, 0], features[row, 1]]) == 1 and features[row, size] == 1:
                    false_negative += 1

            accy = (100 * (true_positive+true_negative)) / features.shape[0]
            if max_acc < accy:
                max_acc = accy
                rgb_prediction = manual_copy
                network_prediction = predict_copy
                original = original_copy

    return max_acc, rgb_prediction, network_prediction, original


def _homogenize(connected_components: np.ndarray,
                network_prediction: np.ndarray,
                rgb_prediction: np.ndarray,
                original: np.ndarray):
    # Imagen en con 0, 1, 2
    result_image = network_prediction.copy()
    # Imagen a color del resultado de la red
    final_image = rgb_prediction.copy()
    img_rgb = original.copy()

    for x in range(1, connected_components[0]):
        mask = connected_components[1] != x
        mask2 = connected_components[1] == x
        result_image_copy = result_image.copy()
        result_image_copy[mask] = 0
        n_veins = np.count_nonzero(result_image_copy == 1)
        n_arteries = np.count_nonzero(result_image_copy == 2)

        if n_veins == 0 and n_arteries == 0:
            pass
        elif n_veins == n_arteries:
            pass
        elif (n_veins == 1 and n_arteries == 0) or (n_veins == 0 and n_arteries == 1):
            pass
        elif n_veins > n_arteries:
            result_image[mask2] = 1
        else:
            result_image[mask2] = 2

    for row in range(0, result_image.shape[0]):
        for col in range(0, result_image.shape[1]):
            if result_image[row, col] == 1:
                final_image[row, col] = [255, 0, 0]
                img_rgb[row, col] = [255, 0, 0]
            elif result_image[row, col] == 2:
                final_image[row, col] = [0, 0, 255]
                img_rgb[row, col] = [0, 0, 255]

    return final_image, img_rgb


def _box_labels(bifurcations: list, c_components: np.ndarray):
    connected_vessels = []
    for b in bifurcations:
        labels = c_components[1]
        box = labels[b[1]-1:b[3]+1, b[0]-1:b[2]+1]
        unique = np.unique(box)
        if len(unique) == 4:
            connected_vessels.append(unique[1:4])
    return connected_vessels


def _average(widths: list):
    acum = 0
    for w in widths:
        acum += w[1]+w[2]
    acum /= len(widths)
    return acum


def _normalize_indexes(connected_matrix: np.ndarray, label: int):
    labeled = connected_matrix[1]
    indexes = np.where(labeled == label)
    formatted_index = []
    for i in range(0, len(indexes[0])):
        formatted_index.append([indexes[0][i], indexes[1][i]])

    return formatted_index


def _average_width(connected_matrix: np.ndarray, connected: list, thr_img: np.ndarray, final_image: np.ndarray):
    connected_avg = []
    for c in connected:
        formatted_indexes = _normalize_indexes(connected_matrix, c)
        label_widths = l.vessel_width(thr_img, formatted_indexes)
        index = int(len(formatted_indexes)/2)
        connected_avg.extend([_average(label_widths), final_image[formatted_indexes[index][0], formatted_indexes[index][1]]])
    return connected_avg


def _coloring(connected_matrix: np.ndarray, box: list, rgb: list, skeleton: np.ndarray):
    for segment_label in box:
        formatted_indexes = _normalize_indexes(connected_matrix, segment_label)
        for index in formatted_indexes:
            skeleton[index[0], index[1]] = rgb
    return skeleton[index[0], index[1]]


def _postprocessing(connected_components: np.ndarray, thr_img: np.ndarray, bifurcs: list, final_img: np.ndarray):
    rgb = final_img.copy()

    connected_vessels = _box_labels(bifurcs, connected_components)
    for triplet in connected_vessels:
        width_and_color = _average_width(connected_components, triplet, thr_img, rgb)
        red = [0, 0]
        blue = [0, 0]
        maxwidth = [-1, -1]

        for i in [1, 3, 5]:
            width = width_and_color[i - 1]
            if width*1.75 > maxwidth[0]:
                maxwidth[0] = width
                maxwidth[1] = width_and_color[i]
            if all(width_and_color[i] == [255, 0, 0]):
                blue[0] += 1
                blue[1] = width
            elif all(width_and_color[i] == [0, 0, 255]):
                red[0] += 1
                red[1] = width

        if (red[0]+blue[0]) == 1:
            pass
        else:
            if not(all(maxwidth[1] == [255, 255, 255])):
                _coloring(connected_components, triplet, maxwidth[1], rgb)

    return rgb


def _accuracy(post_img: np.ndarray, segmented_img: np.ndarray, gt_img: np.ndarray):
    counter = 0
    TN = 0
    FN = 0
    FP = 0
    TP = 0
    for it_x in range(0, segmented_img.shape[0]):
        for it_y in range(0, segmented_img.shape[1]):
            if segmented_img[it_x, it_y] == 255:
                if (all(gt_img[it_x, it_y] == [255, 0, 0]) or all(gt_img[it_x, it_y] == [0, 0, 255])) and \
                        not(all(post_img[it_x, it_y] == [255, 255, 255])):
                    counter += 1
                    if all(post_img[it_x, it_y] == [0, 0, 255]) and all(gt_img[it_x, it_y] == [0, 0, 255]):
                        TP += 1
                    elif all(post_img[it_x, it_y] == [255, 0, 0]) and all(gt_img[it_x, it_y] == [255, 0, 0]):
                        TN += 1
                    elif all(post_img[it_x, it_y] == [0, 0, 255]) and all(gt_img[it_x, it_y] == [255, 0, 0]):
                        FP += 1
                    elif all(post_img[it_x, it_y] == [255, 0, 0]) and all(gt_img[it_x, it_y] == [0, 0, 255]):
                        FN += 1

    sensitivity = TP / (TP + FN)
    specificity = TN / (FP + TN)
    accy = (TP + TN) / (TP + TN + FP + FN)
    return [accy, sensitivity, specificity]


def classification(original_img: np.ndarray, manual_img: np.ndarray):
    manual = manual_img
    bifurcations, crossings = l.classification(manual, 0)
    features, sectioned_img, thr_img, predict_img = _loading_model(original_img, manual, None, 38)
    acc, rgb, network, original = _validating_model(features, sectioned_img, original_img, predict_img, 38, 0)
    connected_components = cv2.connectedComponentsWithStats(sectioned_img.astype(np.uint8), 4, cv2.CV_32S)
    final_img, img_original = _homogenize(connected_components, network, rgb, original)
    post_img = _postprocessing(connected_components, thr_img, bifurcations, img_original)
    return post_img
