import enum
import math
from particle import Pose

class Action(enum.Enum):
  FORWARD=0
  LEFT=1
  BACKWARD=2
  RIGHT=3

class Roomba:
    def __init__(self, pos: Pose, dx: float, dtheta: float) -> None:
        self._pos = pos 
        self._dx = dx
        self._dtheta = dtheta

    def move(self, action: int):
        x, y = self._pos.x, self._pos.y
        theta = self._pos.theta
        action_enum = Action(action)
        if action_enum == Action.FORWARD:
            x += self._dx * math.cos(theta) 
            y += self._dx * math.sin(theta) 
        elif action_enum == Action.LEFT:
            theta += self._dtheta 
        elif action_enum == Action.BACKWARD:
            x -= self._dx * math.cos(theta)
            y -= self._dx * math.sin(theta)
        elif action_enum == Action.RIGHT:
            theta -= self._dtheta
        self._pos = Pose(x=x, y=y, theta=theta)

    @property
    def pose(self):
      return self._pos
