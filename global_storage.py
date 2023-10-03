import queue

# Set this to True to exit
exit_signal = False

# Images from camera
rgb_frames = queue.Queue(5)
depth_frames = queue.Queue(5)
record_videos = False

# Emergency STOP
# When the car hits obstacle or when
# user presses emergency stop button
# following value will be set to True
# causing car to brake
emergency_stop = True


# Motor states
pause = False
speed = 0
steer = 0

# Buttons
button_1 = False
button_2 = False
button_3 = False
button_4 = False
button_ss1 = False
button_ss2 = False

# Led
led_state = False

# Remote control sensor
last_control_signal_time = 0
remote_control_max_speed = 100
remote_control_max_steer_angle = 150
remote_control_speed = 0
remote_control_steer_angle = 0