
import cv2
import numpy as np
import time
# from tensorflow.keras import backend as K


def predict(image, model):

    start_time = time.time()
    height = 256
    # image = cv2.imread('static/images/plate.jpg')

    width = image.shape[1] * height // image.shape[0]

    image = cv2.resize(image, (width, height))

    image[0:5, :] = [0, 0, 0]
    image[height-6:height-1, :] = [0, 0, 0]
    image[:, 0:5] = [0, 0, 0]
    image[:, width-6:width-1] = [0, 0, 0]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    value_mask = cv2.split(hsv)[2]

    gray = cv2.morphologyEx(gray, cv2.MORPH_ERODE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)))

    # threshold = np.sum(np.sum(value_mask, axis=1) /
    #                    value_mask.shape[1], axis=0) / value_mask.shape[0] * 0.9

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # threshold = np.sum(value_mask.flatten()) / len(value_mask.flatten()) * 0.95

    # ret, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    ret, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.medianBlur(thresh, 3)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    angle = angle - 90
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # rotate the image to deskew it
    (h, w) = (height, width)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    thresh = cv2.warpAffine(
        thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    th, thresh = cv2.threshold(
        thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # cv2.imwrite('static/images/debug/thresh.jpg', thresh)
    # _, labels = cv2.connectedComponents(thresh)

    # mask = np.zeros(thresh.shape, dtype="uint8")

    # total_pixels = width * height

    # lower = total_pixels // 140
    # upper = total_pixels // 10

    # for (i, label) in enumerate(np.unique(labels)):
    #     if label == 0:
    #         continue

    #     labelMask = np.zeros(thresh.shape, dtype="uint8")
    #     labelMask[labels == label] = 255
    #     numPixels = cv2.countNonZero(labelMask)
    #     if numPixels > lower and numPixels < upper:
    #         mask = cv2.add(mask, labelMask)

    # cnts, _ = cv2.findContours(
    #     mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    # def compare(rect1, rect2):
    #     if abs(rect1[1] - rect2[1]) > 20:
    #         return rect1[1] - rect2[1]
    #     else:
    #         return rect1[0] - rect2[0]

    # boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare))

    # min_width = np.sum(np.array(boundingBoxes)[
    #                    :, 2]) / len(boundingBoxes) * 0.4
    # max_width = np.sum(np.array(boundingBoxes)[
    #                    :, 2]) / len(boundingBoxes) * 1.8
    # min_height = np.sum(np.array(boundingBoxes)[
    #                     :, 3]) / len(boundingBoxes) * 0.3
    # max_height = np.sum(np.array(boundingBoxes)[
    #                     :, 3]) / len(boundingBoxes) * 1.8

    # tempBoxes = boundingBoxes.copy()
    # for c in boundingBoxes:
    #     x, y, w, h = c
    #     if(w > min_width and w < max_width and h > min_height and h < max_height):
    #         cv2.rectangle(temp_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #     else:
    #         cv2.rectangle(temp_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    #         tempBoxes.remove(c)
    # boundingBoxes = tempBoxes.copy()
    # boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare))

    # Segmentation

    (rows, cols) = thresh.shape
    h_projection = np.sum(thresh, axis=1)
    h_projection = h_projection / np.max(h_projection)
    threshold_h = (np.max(h_projection) -
                   np.min(h_projection[h_projection > 0])) / 6
    cpt = thresh.copy()

    # select the black areas
    black_areas = np.where(h_projection < threshold_h)

    # Set all pixels to 0 if their row sum lower than threshold
    for j in black_areas:
        cpt[j, :] = 0

    lines = []
    line = []
    min_height = len(h_projection) / 9
    cpt = cv2.copyMakeBorder(cpt, 10, 10, 0, 0, cv2.BORDER_CONSTANT)
    h_projection = np.sum(cpt, axis=1)

    for y in range(len(h_projection)-1):
        # if h_projection[y + 1] <= 2 < h_projection[y]:
        if h_projection[y] == 0 and h_projection[y+1] > 0:
            line.append(y)
        if len(line) and h_projection[y+1] == 0 and h_projection[y] > 0:
            if y - line[0] > min_height:
                line.append(y)
                lines.append(cpt[line[0]:line[1], :])
            line = []

    letters = []

    for line in lines:

        # line = cv2.morphologyEx(line, cv2.MORPH_DILATE, cv2.getStructuringElement(
        #     cv2.MORPH_RECT, (2, 2)), iterations=1)

        v_projection = np.sum(line, axis=0)
        v_projection = v_projection / np.max(v_projection)
        threshold_v = (np.max(v_projection) -
                       np.min(v_projection[v_projection > 0])) / 10

        # select the black areas
        black_areas = np.where(v_projection < threshold_v)

        for j in black_areas:
            line[:, j] = 0

        # Add padding to lines
        line = cv2.copyMakeBorder(line, 0, 0, 10, 10, cv2.BORDER_CONSTANT)
        min_width = len(v_projection) // ((18 // len(lines)))

        v_projection = np.sum(line, axis=0)
        letter = []

        letters_line = []

        for y in range(len(v_projection)-1):
            if v_projection[y] == 0 and v_projection[y+1] > 0:
                letter.append(y)
            if len(letter) and v_projection[y+1] == 0 and v_projection[y] > 0:
                if len(v_projection) - y == 4 and v_projection[y+4] > 0:
                    y = y + 4
                if y - letter[0] > min_width and y - letter[0] < len(v_projection)*0.8:
                    letter.append(y)
                    # cv2.imwrite('static/images/debug/'+str(y) +
                    #             '.jpg', line[:, letter[0]:letter[1]])
                    letters_line.append(line[:, letter[0]:letter[1]])

                letter = []

        letters.append(letters_line)

    chars = [
        '0',
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        'ma',
        'ka',
        'sa',
        'ja',
        'na',
        'ba',
        'ga',
        'la',
        'dhha',
        'bha',
        'ra',
        'kha',
        'da',
        'ya',
        'gha',
        'cha',
        'jha',
        'yna',
        'thha',
        'pa',
    ]

    class_indices = {'0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 6, '15': 7, '16': 8, '17': 9, '18': 10, '19': 11, '2': 12, '20': 13, '21': 14,
                     '22': 15, '23': 16, '24': 17, '25': 18, '26': 19, '27': 20, '28': 21, '29': 22, '3': 23, '4': 24, '5': 25, '6': 26, '7': 27, '8': 28, '9': 29}

    plate = ""

    for ll in letters:
        for l in ll:
            crop = l.copy()
            rows, columns = crop.shape

            crop = cv2.morphologyEx(crop, cv2.MORPH_OPEN, cv2.getStructuringElement(
                cv2.MORPH_RECT, (2, 2)), iterations=1)

            crop = cv2.erode(crop, cv2.getStructuringElement(
                cv2.MORPH_RECT, (2, 2)), iterations=2)

            paddingY = (52-rows) // 2 if rows < 52 else int(0.1*rows)
            paddingX = (52-columns) // 2 if columns < 52 else int(0.15*columns)

            crop = cv2.copyMakeBorder(
                crop, paddingY, paddingY, paddingX, paddingX, cv2.BORDER_CONSTANT, None, 0)
            crop = cv2.resize(crop, (52, 52))
            crop = crop.astype('float') / 255.0
            crop = np.reshape(crop, (1, 52, 52, 1))

            prob = model.predict(crop)[0]
            idx = np.argmax(prob, axis=-1)
            char = chars[int(list(class_indices.keys())[
                list(class_indices.values()).index(idx)])]
            plate += ' ' + char
    print('Prediction done in :', time.time()-start_time)
    return (plate, len(letters), time.time()-start_time)
