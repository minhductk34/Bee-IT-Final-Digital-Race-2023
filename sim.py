import json
import cv2
import numpy as np

import asyncio
import websockets
from PIL import Image
import base64
from io import BytesIO

model_traffic = cv2.dnn.readNetFromONNX("traffic_sign_classifier_lenet_v2.onnx")

def filter_signs_by_color(image):
    """Lọc các đối tượng màu đỏ và màu xanh dương - Có thể là biển báo.
        Ảnh đầu vào là ảnh màu BGR
    """
    # Chuyển ảnh sang hệ màu HSV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Lọc màu đỏ cho stop và biển báo cấm
    lower1, upper1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower2, upper2 = np.array([170, 70, 50]), np.array([180, 255, 255])
    mask_1 = cv2.inRange(image, lower1, upper1) # dải màu đỏ thứ nhất
    mask_2 = cv2.inRange(image, lower2, upper2) # dải màu đỏ thứ hai
    mask_r = cv2.bitwise_or(mask_1, mask_2) # kết hợp 2 kết quả từ 2 dải màu khác nhau

    # Lọc màu xanh cho biển báo điều hướng
    lower3, upper3 = np.array([85, 50, 200]), np.array([135, 250, 250])
    mask_b = cv2.inRange(image, lower3,upper3)

    # Kết hợp các kết quả
    mask_final  = cv2.bitwise_or(mask_r,mask_b)
    return mask_final
def get_boxes_from_mask(mask):
    """Tìm kiếm hộp bao biển báo
    """
    bboxes = []

    nccomps = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    numLabels, labels, stats, centroids = nccomps
    im_height, im_width = mask.shape[:2]
    for i in range(numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        # Lọc các vật quá nhỏ, có thể là nhiễu
        if w < 20 or h < 20:
            continue
        # Lọc các vật quá lớn
        if w > 0.8 * im_width or h > 0.8 * im_height:
            continue
        # Loại bỏ các vật có tỷ lệ dài / rộng quá khác biệt
        if w / h > 2.0 or h / w > 2.0:
            continue
        bboxes.append([x, y, w, h])
    return bboxes
def detect_traffic_signs(img, model, draw=None):
    """Phát hiện biển báo
    """

    # Các lớp biển báo
    classes = ['unknown', 'left', 'no_left', 'right',
               'no_right', 'straight', 'stop']

    # Phát hiện biển báo theo màu sắc
    mask = filter_signs_by_color(img)
    bboxes = get_boxes_from_mask(mask)

    # Tiền xử lý
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img / 255.0

    # Phân loại biển báo dùng CNN
    signs = []
    for bbox in bboxes:
        # Cắt vùng cần phân loại
        x, y, w, h = bbox
        sub_image = img[y:y+h, x:x+w]

        if sub_image.shape[0] < 20 or sub_image.shape[1] < 20:
            continue

        # Tiền xử lý
        sub_image = cv2.resize(sub_image, (32, 32))
        sub_image = np.expand_dims(sub_image, axis=0)

        # Sử dụng CNN để phân loại biển báo
        model.setInput(sub_image)
        preds = model.forward()
        preds = preds[0]
        cls = preds.argmax()
        score = preds[cls]

        # Loại bỏ các vật không phải biển báo - thuộc lớp unknown
        if cls == 0:
            continue

        # Loại bỏ các vật có độ tin cậy thấp
        if score < 0.9:
            continue

        signs.append([classes[cls], x, y, w, h])
        
        if draw is not None:
            text = classes[cls] + ' ' + str(round(score, 2))
            cv2.rectangle(draw, (x, y), (x+w, y+h), (0, 255, 255), 4)
            cv2.putText(draw, text, (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return signs

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
speed = 0
steer = 0


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
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
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([170, 110, 255])
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


def car_filter(image):
    hsv_car = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_car = np.array([0, 0, 100])
    upper_car = np.array([170, 100, 255])
    mask_car = cv2.inRange(hsv_car, lower_car, upper_car)
    car_filter = cv2.bitwise_and(image, image, mask=mask_car)
    car_filter = cv2.cvtColor(car_filter, cv2.COLOR_HSV2BGR)
    car_filter = cv2.cvtColor(car_filter, cv2.COLOR_BGR2GRAY)
    return mask_car


def left_right_point(image, hough, mask_car, draw, line=0.85):
    global center, lane_width, left_point, right_point, has_car_left, has_car_right

    car_line_y = int(IMAGE_H * 0.65)
    car_line_y2 = int(IMAGE_H * 0.85)
    red_line_car = [mask_car[car_line_y, :], mask_car[car_line_y2, :]]

    if gostraight_count != 0:
        image = hough

    red_line_y = int(IMAGE_H * line)
    red_line = image[red_line_y, :]

    find_left = find_right = False
    for x in range(center - 15, 0, -1):
        if red_line[x].any() > 0:
            left_point = x
            find_left = True
            break
    for x in range(center + 15, IMAGE_W):
        if red_line[x].any() > 0:
            right_point = x
            find_right = True
            break
    
    if find_left and not find_right:
        right_point = left_point + lane_width
    if find_right and not find_left:
        left_point = right_point - lane_width
    
    center = (right_point + left_point) // 2
    lane_width = right_point - left_point

    has_car_left = False
    has_car_right = False
    for line in red_line_car:
        for x in range(center, IMAGE_W):
            if line[x].all() == 0:
                has_car_right = True
                break
        for x in range(center, 0, -1):
            if line[x].all() == 0:
                has_car_left = True
                break

    cv2.line(draw, (left_point, red_line_y), (right_point, red_line_y), (0, 0, 255), 1)
    cv2.circle(draw, (left_point, red_line_y), 5, (255, 0, 0), -1)
    cv2.circle(draw, (right_point, red_line_y), 5, (0, 255, 0), -1)
    cv2.circle(draw, (center, red_line_y), 5, (0, 0, 255), -1)

    cv2.line(draw, (0, car_line_y), (IMAGE_W, car_line_y), (0, 0, 255), 1)
    cv2.line(draw, (0, car_line_y2), (IMAGE_W, car_line_y2), (0, 0, 255), 1)


def top_right_left_point(hough, draw, line=0.85):
    global top_right_point, top_left_point, turnright_count, turnleft_count, duration, turnleft, turnright, gostraight_count, gostraight

    red_line_y = int(IMAGE_H * line)

    red_line_x2 = left_point - lane_width//3.3
    red_line_x3 = right_point + lane_width//3.3

    red_line_2 = hough[:, red_line_x2]
    red_line_3 = hough[:, red_line_x3]

    distance = 50
    has_right = False
    has_left = False

    for x in range(red_line_y, red_line_y - distance, -1):
        if red_line_2[x].any() > 0:
            top_right_point = x
            has_right = True
            break

    for x in range(red_line_y, red_line_y - distance, -1):
        if red_line_3[x].any() > 0:
            top_left_point = x
            has_left = True
            break

    if has_right and turnright:
        turnright_count = 1
        turnright = False
        duration = 30

    if has_left and turnleft:
        turnleft_count = 1
        turnleft = False
        duration = 30

    if gostraight:
        if has_right or has_left:
            gostraight_count = 1
            gostraight = False
            duration = 20

    cv2.line(draw, (red_line_x2, red_line_y), (red_line_x2, top_right_point), (0, 0, 255), 1)
    cv2.line(draw, (red_line_x3, red_line_y), (red_line_x3, top_left_point), (0, 0, 255), 1)
    cv2.circle(draw, (red_line_x2, top_right_point), 5, (0, 255, 0), -1)
    cv2.circle(draw, (red_line_x3, top_left_point), 5, (255, 0, 0), -1)

def calculate_speed_angle(signs):
    global turnleft_count, turnright_count, turnleft, turnright, gostraight_count, gostraight, duration, speed, steer, stop

    speed = 0.5
    img_centerx = IMAGE_W // 2

    center_point_right = center + lane_width / 4
    center_point_left = center - lane_width / 4

    center_point = center

    center_diff = img_centerx - center_point

    steer = - float(center_diff * 0.03)

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
        speed = 0
        steer = -1
        if turnleft_count >= duration:
            turnleft_count = 0
            duration = 0

    if turnright_count > 0:
        turnright_count += 1
        speed = 0
        steer = 1
        if turnright_count >= duration:
            turnright_count = 0
            duration = 0

    if gostraight_count > 0:
        gostraight_count += 1
        steer = 0
        if gostraight_count >= duration:
            gostraight_count = 0
            duration = 0

    speed -= abs(steer) * 1.3
    if speed <= 0:
        speed = 0.1

    if stop < 0:
        stop += 1
        speed = 0
        if stop >= duration:
            stop = 0
            duration = 0

    return speed, steer

async def echo(websocket, path):
    global IMAGE_H, IMAGE_W
    async for message in websocket:
        # Get image from car
        data = json.loads(message)
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        IMAGE_H, IMAGE_W = image.shape[:2]

        # Birdview transformation and Canny
        image_birdview = birdview(image)
        canny_image = canny(image_birdview)

        # Prepare images to be draw on
        point_image = birdview(image)
        traffic_image = image.copy()
        hough_image = birdview(image)

        # Prepare images for algorithm
        w_filter = white_filter(image)
        hough = houglines(w_filter, hough_image)
        mask_car = car_filter(image)

        # Algorithm
        left_right_point(w_filter, hough, mask_car, point_image)
        #top_right_left_point(hough, point_image)
        signs = detect_traffic_signs(image, model_traffic, traffic_image)


        # Mesure and decide speed and angle
        speed, steer = calculate_speed_angle(signs)

        # Show images
        #cv2.imshow("image", image)
        #cv2.imshow("canny", canny_image)
        #cv2.imshow("white filter", w_filter)
        #cv2.imshow("hough", hough_image)
        cv2.imshow("point", point_image)
        cv2.imshow("traffic", traffic_image)
        #cv2.imshow('mask_car', mask_car)

        cv2.waitKey(1)
        message = json.dumps({"throttle": speed, "steering": steer})
        print(message)
        await websocket.send(message)

async def main():
    async with websockets.serve(echo, "0.0.0.0", 4567, ping_interval=None):
        await asyncio.Future()

asyncio.run(main())
