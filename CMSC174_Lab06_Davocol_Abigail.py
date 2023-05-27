import cv2
import numpy as np

def bg_subtractor(subtractor):
    match subtractor:
        case "mog2":
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        case "knn":
            bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=False)
        case "gmg":
            bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGMG()
        case "cnt":
            bg_subtractor = cv2.bgsegm.createBackgroundSubtractorCNT()
        case "gsoc":
            bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGSOC()
        case "lsbp":
            bg_subtractor = cv2.bgsegm.createBackgroundSubtractorLSBP()

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    cap = cv2.VideoCapture('clock.mp4')
    success, frame = cap.read()
    while success:

        fg_mask = bg_subtractor.apply(frame)

        _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        cv2.erode(thresh, erode_kernel, thresh, iterations=2)
        cv2.morphologyEx(thresh,cv2.MORPH_OPEN, open_kernel,iterations=2)
        cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)

        
        contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > 500:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.intp(box)

                cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

        cv2.imshow(subtractor, fg_mask)
        cv2.imshow('thresh', thresh)
        cv2.imshow('detection', frame)

        k = cv2.waitKey(30)
        if k == 27:  
            break

        success, frame = cap.read()

def main():
    subtractor = input("Choose a subtractor: ")
    bg_subtractor(subtractor)

main()