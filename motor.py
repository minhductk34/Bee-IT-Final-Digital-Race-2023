import global_storage as gs
import config as cf
import cv2
import numpy as np
import _thread

from platform_modules.motor_controller import MotorController
from platform_modules.camera import Camera
from utils.keyboard_getch import _Getch

from onnx import detect_traffic_signs, model

camera = Camera()
camera.start()
motor_controller = MotorController()
motor_controller.start()
getch = _Getch()

center = 160
lane_width = 96
left_point = 100
right_point = 220
top_right_point = 0
top_left_point = 0

turnleft = False
turnright = False
gostraight = False
stop = 0
turnleft_count = 0
turnright_count = 0
gostraight_count = 0
duration = 0

IMAGE_H = 240
IMAGE_W = 320

template1 = cv2.imread("traffic/stop.png", 0)
template4 = cv2.imread("traffic/left.png", 0)
template5 = cv2.imread("traffic/right.png", 0)
templates1 = [template1]
templates2 = [template4, template5]
texts1 = ["stop"]
texts2 = ["left", "right"]


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    canny = cv2.Canny(blur, 30, 80)
    return canny


def birdview(image):
    src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, IMAGE_H // 2], [IMAGE_W, IMAGE_H // 2]])
    dst = np.float32([[100, IMAGE_H], [IMAGE_W - 100, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped_image = cv2.warpPerspective(image, M, (IMAGE_W, IMAGE_H))
    return warped_image


def white_filter(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 255])
    upper_white = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    white_filter = cv2.bitwise_and(image, image, mask=mask)
    white_filter = cv2.cvtColor(white_filter, cv2.COLOR_HSV2BGR)
    white_filter = cv2.cvtColor(white_filter, cv2.COLOR_BGR2GRAY)
    return mask


def houglines(image, draw):
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, 30, minLineLength=3, maxLineGap=150)
    hough = np.zeros([IMAGE_H, IMAGE_W, 3])
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(hough, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return hough


def left_right_point(image, draw, line=0.85):
    global center, lane_width, left_point, right_point

    red_line_y = int(IMAGE_H * line)
    red_line = image[red_line_y, :]

    find_left = False
    find_right = False

    for x in range(center - 10, 0, -1):
        if red_line[x].all() > 0:
            left_point = x
            find_left = True
            break
    for x in range(center + 10, IMAGE_W):
        if red_line[x].all() > 0:
            right_point = x
            find_right = True
            break

    if not find_left and find_right:
        left_point = right_point - lane_width
    if not find_right and find_left:
        right_point = left_point + lane_width

    center = (right_point + left_point) // 2
    lane_width = right_point - left_point

    cv2.line(draw, (left_point, red_line_y), (right_point, red_line_y), (0, 0, 255), 1)
    cv2.circle(draw, (left_point, red_line_y), 5, (255, 0, 0), -1)
    cv2.circle(draw, (right_point, red_line_y), 5, (0, 255, 0), -1)
    cv2.circle(draw, (center, red_line_y), 5, (0, 0, 255), -1)


def filter_signs_by_color(image):
    # RED
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower1, upper1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower2, upper2 = np.array([170, 70, 50]), np.array([180, 255, 255])
    mask_1 = cv2.inRange(image, lower1, upper1)
    mask_2 = cv2.inRange(image, lower2, upper2)
    mask_r = cv2.bitwise_or(mask_1, mask_2)
    # BLUE
    lower3, upper3 = np.array([100, 100, 0]), np.array([140, 255, 255])
    mask_b = cv2.inRange(image, lower3, upper3)
    # COMBINE
    mask_final = cv2.bitwise_or(mask_r, mask_b)
    return mask_r, mask_b


def get_boxes_from_mask(mask_r, mask_b):
    bboxes_r = []
    bboxes_b = []
    nccomps = cv2.connectedComponentsWithStats(mask_r, 4, cv2.CV_32S)
    numLabels, labels, stats, centroids = nccomps
    im_height, im_width = mask_r.shape[:2]
    for i in range(numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if w + h < 30:
            continue
        if w > 0.8 * im_width or h > 0.8 * im_height:
            continue
        if w / h > 1.4 or h / w > 1.4:
            continue
        bboxes_r.append([x, y, w, h])
    nccomps = cv2.connectedComponentsWithStats(mask_b, 4, cv2.CV_32S)
    numLabels, labels, stats, centroids = nccomps
    im_height, im_width = mask_b.shape[:2]
    for i in range(numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if w + h < 30:
            continue
        if w > 0.8 * im_width or h > 0.8 * im_height:
            continue
        if w / h > 1.4 or h / w > 1.4:
            continue
        bboxes_b.append([x, y, w, h])
    return bboxes_r, bboxes_b


def detect_traffic_signs_without_model(img, draw=None):
    mask_r, mask_b = filter_signs_by_color(img)
    bboxes_r, bboxes_b = get_boxes_from_mask(mask_r, mask_b)
    list_text = []

    # RED SIGNS
    for bbox in bboxes_r:
        x, y, w, h = bbox
        list_value = []
        crop_img = img[y:y + h, x:x + w]
        crop_img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        for template in templates1:
            crop_template = cv2.resize(template, (w, h))

            result = cv2.matchTemplate(crop_img_gray, crop_template, cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            list_value.append(max_val)
        max_value = max(list_value)
        text = ""
        if max_value > 800000:
            text = texts1[list_value.index(max_value)]
        list_text.append(text)
        if draw is not None and text != "":
            cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(draw, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # BLUE SIGNS
    for bbox in bboxes_b:
        x, y, w, h = bbox
        list_value = []
        crop_img = img[y:y + h, x:x + w]
        crop_img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        for template in templates2:
            crop_template = cv2.resize(template, (w, h))
            result = cv2.matchTemplate(crop_img_gray, crop_template, cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            list_value.append(max_val)
        max_value = max(list_value)
        text = ""
        if max_value > 700000:
            text = texts2[list_value.index(max_value)]
        list_text.append(text)
        if draw is not None and text != "":
            cv2.rectangle(draw, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(draw, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return list_text


def calculate_speed_angle(signs):
    speed = 0.3
    center_diff = IMAGE_W // 2 - center
    steer = -float(center_diff * 0.04)

    speed -= abs(steer) * 1.3
    if speed <= 0.12:
        speed = 0.12
        
    if not turnleft and not turnright and not gostraight:
        if "left" in signs:
            turnleft = True
        if "right" in signs:
            turnright = True
        if "straight" in signs:
            gostraight = True
    if "stop" in signs:
        stop = 1
        duration = 100
    if turnleft_count > 0:
        turnleft_count += 1
        speed = 0.12
        steer = -1
        if turnleft_count >= duration:
            turnleft_count = 0
            duration = 0
    if turnright_count > 0:
        turnright_count += 1
        speed = 0.12
        steer = 1
        if turnright_count >= duration:
            turnright_count = 0
            duration = 0
    if gostraight_count > 0:
        gostraight_count += 1
        steer = 0.12
        if gostraight_count >= duration:
            gostraight_count = 0
            duration = 0  
    if stop > 0:
        stop += 1
        speed = 0
        if stop >= duration:
            stop = 0
            duration = 0

    return speed, steer


def keyboard():
    key = getch()
    if key == 'q':
        gs.exit_signal = True

_thread.start_new_thread(keyboard(), ())

def get_image_from_camera():
    if not gs.rgb_frames.empty():
        rgb = gs.rgb_frames.get()
    if not gs.depth_frames.empty():
        depth = gs.depth_frames.get()
    return rgb, depth


while not gs.exit_signal:
    # Take image from camera
    rgb, depth = get_image_from_camera()
    image = cv2.flip(rgb, 1)

    # Birdview transformation and Canny
    image_birdview = birdview(image)
    canny_image = canny(image)

    # Prepare images to be draw on
    point_image = birdview(image)
    traffic_image = image.copy()
    hough_image = birdview(image)

    # Prepare images for algorithm
    w_filter = white_filter(image)
    #hough = houglines(w_filter, hough_image)

    # Algorithm
    left_right_point(canny_image, point_image)
    #signs = detect_traffic_signs_without_model(image, traffic_image)
    signs = detect_traffic_signs(image, model, traffic_image)

    # Mesure and decide speed and angle
    speed, steer = calculate_speed_angle(signs)
    
    gs.speed = speed * 50
    gs.steer = steer * 60
    if gs.speed > cf.MAX_SPEED:
        gs.speed = cf.MAX_SPEED
    if gs.steer > cf.MAX_ANGLE:
        gs.steer = cf.MAX_ANGLE
    if gs.steer < -cf.MAX_ANGLE:
        gs.steer = -cf.MAX_ANGLE
    
    # Show images
    #cv2.imshow("image", image)
    cv2.imshow("canny", canny_image)
    #cv2.imshow("white filter", w_filter)
    #cv2.imshow("hough", hough_image)
    cv2.imshow("point", point_image)
    cv2.imshow("traffic", traffic_image)
    #cv2.imshow('mask_car', mask_car)

    cv2.waitKey(1)
    print("speed: " + str(gs.speed) + " steer: " + str(gs.steer))

motor_controller.join()
camera.join()
