import cv2
import imutils
import numpy as np
import glob
import re
import pytesseract

cv_img = []
total_img = 0
right_img = 0
custom_config = r'--oem 3 --psm 6'

def detect_plate(img_path):

    # Param
    max_size = 5000
    min_size = 900

    # Load image

    # Resize image
    img = cv2.resize(img_path, (620, 480))

    # Edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Blur to reduce noise
    edged = cv2.Canny(gray, 30, 200)  # Perform Edge detection

    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    screenCnt = None

    # loop over our contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)

        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4 and max_size > cv2.contourArea(c) > min_size:
            screenCnt = approx
            break

    if screenCnt is None:
        detected = 0
        print ("No plate detected: ")
    else:
        detected = 1

    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

        # Masking the part other than the number plate
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
        new_image = cv2.bitwise_and(img, img, mask=mask)

        # Now crop
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

        plate_detected = pytesseract.image_to_string(Cropped, config=custom_config)
        plate_detected = re.sub('[^A-Za-z0-9]+', '', plate_detected)
        return plate_detected
        # Display image
        # cv2.imshow('Input image', img)
        # cv2.imshow('License plate', Cropped)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

for img in glob.glob("plate/*.jpg"):
    img_name = ''
    n= cv2.imread(img,cv2.IMREAD_COLOR)
    # plate detected by tesseract
    plate_detected = detect_plate(n)

    # name of the file
    img_name = re.sub('[^A-Za-z0-9]+', '', img)
    img_name = img_name.replace("plate", "")
    img_name = img_name.replace("jpg", "")

    total_img = total_img +1
    if(img_name == plate_detected):
        right_img = right_img + 1
    
   
    print('img real name: ', img_name, ', img detected name: ', plate_detected, ', total numb of imgs: ', total_img, ', right img: ', right_img)
