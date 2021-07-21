import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.colors as colors
import numpy as np
import time
import cv2


def color_predict(image):

    # RGB Image
    # image = img.imread('static/images/plate.jpg')
    start_time = time.time()
    image = cv2.resize(image, (128, 32))
    # HSV Image
    image_hsv = colors.rgb_to_hsv(image / 255)
    # Color lists
    r = []
    g = []
    b = []
    h = []
    s = []
    v = []

    # RGB Plot
    for line in image:
        for pixel in line:
            temp_r, temp_g, temp_b = pixel
            r.append(temp_r)
            g.append(temp_g)
            b.append(temp_b)

    # HSV Plot
    for line in image_hsv:
        for pixel in line:
            temp_h, temp_s, temp_v = pixel
            h.append(temp_h)
            s.append(temp_s)
            v.append(temp_v)

    df = pd.DataFrame({'red': r, 'blue': b, 'green': g,
                       'hue': h, 'saturation': s, 'value': v})

    kmeans = KMeans(n_clusters=3)

    # RGB Clustering
    kmeans.fit(df[['red', 'green', 'blue']])
    rgb_kmeans = kmeans.predict(df[['red', 'green', 'blue']])
    rgb_centers = kmeans.cluster_centers_

    kmeans = KMeans(n_clusters=3)
    # HSV Clustering
    kmeans.fit(df[['hue', 'saturation', 'value']])
    hsv_kmeans = kmeans.predict(df[['hue', 'saturation', 'value']])
    hsv_centers = kmeans.cluster_centers_

    rgb_captured = []
    for center in rgb_centers:
        c_r, c_g, c_b = center
        rgb_captured.append((
            c_r / 255,
            c_g / 255,
            c_b / 255))

    hsv_captured = []
    for center in hsv_centers:
        c_h, c_s, c_v = colors.hsv_to_rgb(center)
        hsv_captured.append((
            c_h,
            c_s,
            c_v
        ))

    lower_red = [90, 0, 0]
    upper_red = [255, 100, 100]

    lower_green = [0, 80, 0]
    upper_green = [100, 255, 100]

    lower_black = [0, 0, 0]
    upper_black = [100, 100, 100]
    color = ''
    for x in hsv_captured:
        cls = np.array(x) * 255
        if (cls[0] > lower_red[0] and cls[0] < upper_red[0]) and (cls[1] > lower_red[1] and cls[1] < upper_red[1]) and (cls[2] > lower_red[2] and cls[2] < upper_red[2]):
            color = 'red'
            break

        if (cls[0] > lower_green[0] and cls[0] < upper_green[0]) and (cls[1] > lower_green[1] and cls[1] < upper_green[1]) and (cls[2] > lower_green[2] and cls[2] < upper_green[2]):
            color = 'green'
            break

    if color == '':
        for x in hsv_captured:
            cls = np.array(x) * 255
            if (cls[0] > lower_black[0] and cls[0] < upper_black[0]) and (cls[1] > lower_black[1] and cls[1] < upper_black[1]) and (cls[2] > lower_black[2] and cls[2] < upper_black[2]):
                color = 'black'
                break
    print('Color prediction done in :', time.time()-start_time)
    return (color, time.time()-start_time)
