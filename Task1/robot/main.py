import json
import math
import os
import subprocess
import sys
import time
import cv2
import mujoco
import mujoco.viewer
import robot_vision

DATA_FILE = "distances_live.json" #файлик для обмена данными с live_plot.py

model = mujoco.MjModel.from_xml_path("scene_w_robot.xml") #постоянные сцены: массы размеры и тп
data = mujoco.MjData(model) #переменные сцены: скорости расстояния и тп

#управление колсесами робота
left = model.actuator("left_wheel").id 
right = model.actuator("right_wheel").id


chassis_id = model.body("chassis").id
qadr = model.jnt_qposadr[model.body_jntadr[chassis_id]]
#адреса вектора qpos, где лежат координаты и ориентация корпуса робота (x, y, z, qw, qx, qy, qz)

#тут все что связано с шарами
BALL_BODY_NAMES = {"red": "ball_red", "yellow": "ball_yellow", "green": "ball_green", "blue": "ball_blue"}
ball_qadr = {
    name: model.jnt_qposadr[model.body_jntadr[model.body(body_name).id]]
    for name, body_name in BALL_BODY_NAMES.items()
}
#тоже адреса вектора qpos, где лежат координаты и ориентация шариков (x, y, z, qw, qx, qy, qz)

history = {name: {"t": [], "true": [], "vision": []} for name in BALL_BODY_NAMES}
#тут хранятся расстояния до шариков по камере и по факту


CAM_HEIGHT, CAM_WIDTH = 480, 640 #тут разрешение камеры
BALL_DIAMETER = 0.10  #тут реальный диаметр шарика, м
PRINT_PERIOD = 0.05   #период обновления графика


cam_id = model.camera("front_cam").id
fovy = model.cam_fovy[cam_id] #вертикальный угол обзора
focal_px = (CAM_HEIGHT / 2) / math.tan(math.radians(fovy) / 2) #фокусное расстояне в пикселях 

CAMERA_MOUNT_TILT = math.radians(15)  #угол наклона камеры на роботе (см. xyaxes камеры в XML)
CAM_HEIGHT_NOMINAL = 0.7              #заданная высота камеры над полом из модели(см хмлку)


renderer = mujoco.Renderer(model, CAM_HEIGHT, CAM_WIDTH) #"фоткает" сцену с камеры в симуляции, возвращает RGB массив пиксеоей
next_print_time = 0.0 #счетчик обновлений данных для графика 
pitch_estimate = 0.0  #угол наклона корпуса в начале, потом интегрируется угловая скорость гироскопа

V_FWD = 2.2   #скорость колёс на прямых участках, рад/с
V_TURN = 0.6  #скорость колёс при развороте на месте, рад/с
N = 0.6       #длина каждого прямого участка, м
MAX_ACCEL = 1.0  #максимальное изменение целевой скорости колеса, рад/с^2, нужно чтобы робот двигался плавно 

#здесь написаны "фазы" движения робота: он катается вперед и назад по прямой, не разворачиваясь 
PHASES = [
    ("drive", +1), #тут вообще на стадии тестировки были разные варианты траекторий
    ("drive", -1), #еще есть команда "turn", которая разворачивает робота на месте на заданный угол 
]                  #в аргумент поворота передается угол



state = {"idx": 0, "start_xy": None, "start_yaw": None}
#текущее состояние симуляции, фаза движения, старт и угол начала разворота

def yaw_of(data):
   w, x, y, z = data.qpos[qadr + 3: qadr + 7]
   return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
#эта функция использовалась для подсчета поворота робота вокруг своей оси при поворотоах 

#разница двух углов, чтобы между 359 и 1 было 2 градуса, а не 358
def angle_diff(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


#тут управление колесами, вызывается на каждом шаге симуляции физики
def control(model, data):
    kind, param = PHASES[state["idx"] % len(PHASES)]
    x, y = data.qpos[qadr], data.qpos[qadr + 1]

    if kind == "drive":
        sign = 1 if param > 0 else -1
        l = r = sign * V_FWD
        if state["start_xy"] is None:
            state["start_xy"] = (x, y)
        sx, sy = state["start_xy"]
        if math.hypot(x - sx, y - sy) >= N:
            state["idx"] += 1
            state["start_xy"] = None
    else:  # turn
        sign = 1 if param > 0 else -1
        l, r = -sign * V_TURN, sign * V_TURN
        if state["start_yaw"] is None:
            state["start_yaw"] = yaw_of(data)
        if abs(angle_diff(yaw_of(data), state["start_yaw"])) >= math.radians(abs(param)):
            state["idx"] += 1
            state["start_yaw"] = None

    #плавно подводим фактическую скорость к заданной, не больше чем на макс дельта за один шаг симуляции
    max_delta = MAX_ACCEL * model.opt.timestep
    data.ctrl[left] += max(-max_delta, min(max_delta, l - data.ctrl[left]))
    data.ctrl[right] += max(-max_delta, min(max_delta, r - data.ctrl[right]))
    #если сильно "трясти" робота то портятся вычисления на гироскопе и график не попадает  




#наконец запуск симуляции и построения графика
mujoco.set_mjcb_control(control)

#график рисуется в отдельном процессе обычным пайтоном, матплотлиб 
plot_proc = subprocess.Popen([sys.executable, "live_plot.py"])

wall_start = time.time()
#нужно чтобы сопоставить время в муджоко и на графике 

#запуск бесконечной симуляции 
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        
        mujoco.mj_step(model, data) #шаг физики 
        viewer.sync()

        #интегрируем гироскоп, читаем скорость с сенсора и прибавляем к углу наклона корпуса
        wy = data.sensor("chassis_gyro").data[1]
        pitch_estimate += wy * model.opt.timestep

        ahead = data.time - (time.time() - wall_start)
        if ahead > 0:
            time.sleep(ahead)
        #тормозит физику, чтобы не улетала вперед

        #раз в период симуляции делаем шаг, обновляем график и сохраняем данные в json
        if data.time >= next_print_time:
            next_print_time = data.time + PRINT_PERIOD

            renderer.update_scene(data, camera="front_cam") #делает фото сцены с камеры робота
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR) #меняет формат кадра 
            diameters = robot_vision.get_diameters(frame) #достает диаметры с картинки 
            print(robot_vision.get_ball_or_not(frame)) #определяет шарик или не шарик (захотелось начать определять форму объектов)

            cam_pos = data.cam_xpos[cam_id]  #примерно 0.7 м над землей, уже с учётом наклона и позы робота, тут уже точно

            #текущий наклон камеры = штатный наклон крепления +/- наклон всего робота
            total_tilt = CAMERA_MOUNT_TILT + pitch_estimate

            for name, ball_info in diameters.items():
                ball_pos = data.qpos[ball_qadr[name]: ball_qadr[name] + 3]
                true_dist = math.dist(cam_pos, ball_pos)
                #рассчет расстояния по координатам в сцене. 

                
                if ball_info is not None: #если шары вообще есть
                    d_px, center_col, center_row = ball_info #инфа по шару

                    #тут учитывается смещение по вертикали и по горизонтали от гл.о.о. камеры
                    du = (center_col - CAM_WIDTH / 2) / focal_px   #смещение от оси по горизонтали
                    dv = (center_row - CAM_HEIGHT / 2) / focal_px  #смещение от оси по вертикали

                    ray_x = math.cos(total_tilt) - dv * math.sin(total_tilt)    #вперёд
                    ray_y = -du                                                 #вбок, тут просто, робот вбок не заваливается
                    ray_z = -math.sin(total_tilt) - dv * math.cos(total_tilt)  # вниз (отрицательно)
                    ray_len = math.sqrt(ray_x**2 + ray_y**2 + ray_z**2)

                    #суть в том, что строится луч из пикселя с учетом наклона камеры
                    #потом находится пересечение этим лучем пола

                    vision_dis = (CAM_HEIGHT_NOMINAL - BALL_DIAMETER / 2) * ray_len / (-ray_z)
                    
                else: #шаров нет(
                    vision_dist = None
                #складывает время, фактическое расстояние и рассчетное
                history[name]["t"].append(data.time)
                history[name]["true"].append(true_dist)
                history[name]["vision"].append(vision_dist)


            
            #перепись информации сначала во временный файл, потом в считываемый
            #без промежуточных состояний считываемого файла. 
            tmp_json = DATA_FILE + ".tmp"
            with open(tmp_json, "w") as f:
                json.dump(history, f)
            os.replace(tmp_json, DATA_FILE)

plot_proc.terminate() #закрытие графика вместе с симуляцией
