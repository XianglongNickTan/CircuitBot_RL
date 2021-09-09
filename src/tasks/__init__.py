from tasks.clear_one_obstacle import ClearOneObstacle
from tasks.clear_obstacles import ClearObstacles
from tasks.cameras import RealSenseD415
from tasks.pickandplacetask import PickAndPlaceTask
from tasks.construct_bridge import ConstructBridge
from tasks.task import Task

names = {
	'clear-one-obstacle': ClearOneObstacle,
	'clear-obstacles': ClearObstacles,
	'construct-bridge': ConstructBridge
}
