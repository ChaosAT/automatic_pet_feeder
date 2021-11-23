import time
from robot import Robot
from servertest import Server
import threading as th

def main_loop(robot):
    cat = 0
    dog = 1
    while True:
        if robot.CheckMovement():  # 如果检测到位移
            cat_pot_status, dog_pot_status = robot.CheckFood()  #
            cat_time_status, dog_time_status = robot.CheckTime()  #
            cat_status = cat_pot_status and cat_time_status
            dog_status = dog_pot_status and dog_time_status

            if (not cat_status) and (not dog_status):
                print("No need to feed, back to sound detect.")
                continue
            else:
                result_objects = robot.CheckObject()

                if (cat in result_objects) and cat_status:
                    robot.ResupplyFood(cat)
                    cat_pot_status, _ = robot.CheckFood()
                    if cat_pot_status:
                        print("Add Failed, maybe the foodcontainer is empty.")
                    else:
                        robot.SaveTimeRecord('cat')

                if (dog in result_objects) and dog_status:
                    robot.ResupplyFood(dog)
                    _, dog_pot_status = robot.CheckFood()
                    if dog_pot_status:
                        print("Add Failed, maybe the foodcontainer is empty.")
                    else:
                        robot.SaveTimeRecord('dog')
        time.sleep(2)  # scan for every 2 seconds

server = Server()
print("Sever builded.")
robot = Robot()
print("Robot inited.")


th_robot_loop = th.Thread(target=main_loop, args=(robot,))
th_server_loop = th.Thread(target=server.Run, args=(robot,))

th_robot_loop.start()
print("Robot Running.")
th_server_loop.start()
print("Server Running.")
