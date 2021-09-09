from tasks.clear_one_obstacle import ClearOneObstacle
from tasks.clear_obstacles import ClearObstaclesTask
from tasks.cameras import RealSenseD415
from tasks.pickandplacetask import PickAndPlaceTask
from tasks.task import Task

names = {
    'clear-one-obstacle': ClearOneObstacle,
    'clear-obstacles': ClearObstaclesTask,
    'pick-and-place': PickAndPlaceTask
}