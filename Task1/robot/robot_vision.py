import cv2
import mujoco

def masks_from_frame(frame): 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask_red = cv2.inRange(hsv, (0, 150, 0), (10, 255, 255))
    mask_yellow = cv2.inRange(hsv, (10, 150, 0), (35, 255, 255))
    mask_green = cv2.inRange(hsv, (60, 150, 0), (75, 255, 255))
    mask_blue = cv2.inRange(hsv, (110, 150, 0), (120, 255, 255))
    
    masks = {"red": mask_red, "yellow": mask_yellow, "green": mask_green, "blue": mask_blue}

    return masks


def sizes_from_frame(mask):
    connectivity = 4
    output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
    stats = output[2]
    diameter = (stats[1, cv2.CC_STAT_HEIGHT]+stats[1, cv2.CC_STAT_WIDTH])//2
    center_col = stats[1, cv2.CC_STAT_LEFT] + stats[1, cv2.CC_STAT_WIDTH]//2
    center_row = stats[1, cv2.CC_STAT_TOP] + stats[1, cv2.CC_STAT_HEIGHT]//2
    return diameter, center_col, center_row


def get_diameters(frame):
    return {name: sizes_from_frame(mask) if mask.any() else None for name, mask in masks_from_frame(frame).items()}


def is_it_ball(mask):
    connectivity = 4
    output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
    stats = output[2]
    diameter = (stats[1, cv2.CC_STAT_HEIGHT]+stats[1, cv2.CC_STAT_WIDTH])//2
    area = stats[1, cv2.CC_STAT_AREA]
    koef = area/((diameter)**2)
    if koef > 0.7 and koef < 0.9 :
        return "Probably it's a ball"
    else:
        return "Not a ball"


def get_ball_or_not(frame):
    return {name: is_it_ball(mask) if mask.any() else None for name, mask in masks_from_frame(frame).items()}





if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path("scene_w_robot.xml")
    data = mujoco.MjData(model)
    cam = mujoco.Renderer(model, 480, 640)

    while (True):
        mujoco.mj_step(model, data)
        cam.update_scene(data, camera="front_cam")
        frame = cv2.cvtColor(cam.render(), cv2.COLOR_RGB2BGR)

        diameters = get_diameters(frame)
        #print("red: ", diameters["red"])
        #print("yellow: ", diameters["yellow"])
        #print("green: ", diameters["green"])
        #print("blue: ", diameters["blue"])

        cv2.imshow("frame", frame)

        key = cv2.waitKey(80) & 0xFF

        if key == 27: #ESC
            break

    cv2.destroyAllWindows
    cv2.waitKey(10)
