from tasks.clear_one_obstacle import ClearOneObstacle
from tasks.clear_obstacles import ClearObstacles
from tasks.cameras import RealSenseD415
from tasks.pickandplacetask import PickAndPlaceTask
from tasks.construct_one_bridge import ConstructOneBridge
from tasks.construct_bridges import ConstructBridges
from tasks.all_in_one import AllInOne
from tasks.task import Task

names = {
	'clear-one-obstacle': ClearOneObstacle,
	'clear-obstacles': ClearObstacles,
	'construct-one-bridge': ConstructOneBridge,
	'construct-bridges': ConstructBridges,
	'all-in-one': AllInOne
}
