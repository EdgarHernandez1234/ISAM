import rover_learner.camera_provider as cp

def test_build_pipeline_contains_nvargus():
    s = cp.build_csi_gstreamer_pipeline(sensor_id=0, width=640, height=480, fps=30, flip_method=0)
    assert "nvarguscamerasrc" in s
    assert "sensor-id=0" in s
    assert "width=(int)640" in s
    assert "height=(int)480" in s
    assert "framerate=(fraction)30/1" in s
