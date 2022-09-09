
import math

class Sensor:
    def __init__(self, detection_threshold, tol=0.1):
        self._threshold = detection_threshold
        self._tol = tol
        self._angle_delta_left = math.pi / 4
        self._angle_delta_center = 0
        self._angle_delta_right = -math.pi / 4

    def sense(self, roomba, particles):
        readings = particles.get_readings(roomba.pose)
        theta = roomba.pose.theta
        min_left = float('+inf')
        min_center = float('+inf')
        min_right = float('+inf')
        for angle, distance in readings:
            if abs(
                angle - (theta + self._angle_delta_center)
            ) < self._tol and distance < min_center:
                min_center = distance
            elif abs(
                angle - (theta + self._angle_delta_left)
            ) < self._tol and distance < min_left:
                min_left = distance
            elif abs(
                angle - (theta + self._angle_delta_right)
            ) < self._tol and distance < min_right:
                min_right = distance
        return (
                min(min_left, self._threshold) / self._threshold,
                min(min_center, self._threshold) / self._threshold,
                min(min_right, self._threshold) / self._threshold,
        )
            

