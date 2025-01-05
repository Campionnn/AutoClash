import time
import cv2
import numpy as np

def filter_image_by_bgr(image, bgr_values, tolerance=3):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for bgr in bgr_values:
        lower_bound = np.array([max(0, c - tolerance) for c in bgr], dtype=np.uint8)
        upper_bound = np.array([min(255, c + tolerance) for c in bgr], dtype=np.uint8)
        mask |= cv2.inRange(image, lower_bound, upper_bound)
    return cv2.bitwise_and(image, image, mask=mask)


def find_clash(image):
    click_perfect = (235, 186, 47)
    click_good = (128, 57, 32)
    click_bad = (43, 18, 11)
    pointer1 = (16, 5, 93)
    pointer2 = (117, 108, 196)
    pointer3 = (106, 93, 159)
    pointer4 = (2, 0, 55)
    bgr_values = [click_perfect, click_good, click_bad, pointer1, pointer2, pointer3, pointer4]
    filtered_image = filter_image_by_bgr(image, bgr_values, tolerance=3)

    # Preprocess the image
    kernel = np.ones((4, 4), np.uint8)
    filtered_image = cv2.morphologyEx(filtered_image, cv2.MORPH_OPEN, kernel)
    filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    _, filtered_image = cv2.threshold(filtered_image, 1, 255, cv2.THRESH_BINARY)

    # Dilate and contour the image
    kernel = np.ones((5, 5), np.uint8)
    filtered_image = cv2.dilate(filtered_image, kernel, iterations=4)
    contours, _ = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area and return correct one
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(contours) > 0:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 5.4 < w / h < 5.8:
                return x, y, w, h
    return None


start = time.time()
for i in range(2936, 3474, 1):
    image = cv2.imread(f"clash01/frame_{i}.png")
    print(find_clash(image))
print(time.time() - start)
print((3474 - 2936) / (time.time() - start))