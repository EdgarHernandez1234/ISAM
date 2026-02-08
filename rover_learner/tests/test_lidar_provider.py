import math
from rover_learner.lidar_provider import LaserScanLike, min_distance_from_scan

def test_min_distance_forward_cone_filters_invalid():
    # angles from -90 to +90 deg, 181 samples (1 deg increment)
    angle_min = -math.pi/2
    inc = math.pi/180
    ranges = [float('inf')]*181
    # Put a valid close obstacle at +5 deg
    idx = int((math.radians(5) - angle_min)/inc)
    ranges[idx] = 0.8
    # Put an even closer obstacle at +40 deg (outside 15 deg cone)
    idx2 = int((math.radians(40) - angle_min)/inc)
    ranges[idx2] = 0.2

    scan = LaserScanLike(angle_min=angle_min, angle_increment=inc, ranges=ranges, range_min=0.1, range_max=10.0)
    d = min_distance_from_scan(scan, forward_half_angle_deg=15.0)
    assert abs(d - 0.8) < 1e-6

def test_min_distance_none_when_no_valid():
    scan = LaserScanLike(angle_min=0.0, angle_increment=1.0, ranges=[0.0, float('nan'), float('inf')], range_min=0.1, range_max=5.0)
    assert min_distance_from_scan(scan) is None
