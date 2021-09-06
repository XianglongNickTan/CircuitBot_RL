from map_env import MapEnv
import cv2
import pybullet as p


my_map = MapEnv()
my_map.reset()

# my_map._show_3d_weightmap_path()

# action = [39, 19, 37, 28,  1]
# #
# my_map.step(action)

for _ in range(100):
    action = my_map.action_space.sample()
    # print(action)
    my_map.step(action)

print("---------------")
print("done")
print("---------------")

for _ in range(100000):
    p.stepSimulation()
    # my_map.render()
# print(a)