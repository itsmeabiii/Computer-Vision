import cv2
import numpy as np

def tracker(type_tracker):
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    video = cv2.VideoCapture('clock.mp4')
    success, frame = video.read()
    # Define an initial tracking window in the center of the frame.
    # w = 17
    # h = 350
    # x = 690
    # y = 90
    # w = 200
    # h = 300
    # x = 460
    # y = 280

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    while success:

        fg_mask = bg_subtractor.apply(frame)
        _, thresholded_mask = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        cv2.erode(thresholded_mask, erode_kernel, thresholded_mask, iterations=2)
        cv2.morphologyEx(thresholded_mask,cv2.MORPH_OPEN, open_kernel,iterations=2)
        cv2.dilate(thresholded_mask, dilate_kernel, thresholded_mask, iterations=2)

        contours, hier = cv2.findContours(thresholded_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > 500:
                x,y,w,h = cv2.boundingRect(c)
                print(cv2.contourArea(c))


        roi = frame[y:y + h, x:x + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Define the termination criteria:
        # 10 iterations or convergence within 1-pixel radius.
        term_crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)
            
        # Calculate the normalized HSV histogram of the current window.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        backProject = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        match type_tracker:
            case "meanshift":
                # Perform tracking with MeanShift.
                num_iters, track_window = cv2.meanShift(backProject, (x,y,w,h), term_crit)
                # Draw the tracking window.
                x, y, w, h = track_window
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            case "camshift":
                ret, track_window = cv2.CamShift(backProject, (x,y,w,h), term_crit)

                pts = cv2.boxPoints(ret)
                pts = np.intp(pts)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            case "meanshiftK":
                pass
            case "boosting":
                pass
            case "mil":
                pass
            case "kcf":
                pass
            case "tld":
                pass
            case "medianflow":
                pass
            case "goturn":
                pass
            case "mosse":
                pass
            case "csrt":
                pass

        cv2.imshow('thresh', thresholded_mask)
        cv2.imshow('background', backProject)
        cv2.imshow(type_tracker, frame)
        cv2.imshow('roi', roi)

        k = cv2.waitKey(1)
        if k == 27:  # Escape
            break


    video.release()
    cv2.destroyAllWindows()

def main():
    opt_tracker = input("Choose a tracker: ")
    tracker(opt_tracker)

main()