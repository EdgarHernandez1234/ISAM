# ROS 2 Bag Recording Guide for the NVIDIA Jetson

This guide is for the computer science team using the NVIDIA Jetson during rover demos, simulation runs, and debugging sessions.

ROS 2 bag recording is a **software-side capture tool**. It records ROS data traffic so a run can be inspected or replayed later. For this project, bag files are stored on the SSD so the Jetson’s internal storage does not fill up as quickly.

## Purpose

Use ROS 2 bag recording when you want to:

- preserve a milestone demo
- capture a failed run for debugging
- save sensor/controller data for later replay
- keep evidence for technical reporting or validation

Do **not** treat ros2bags as the main user-facing deliverable for the mechanical engineers. Their primary outputs should still be the normal project artifacts such as `demo_artifacts`, videos, logs, and CSV outputs.

---

## Storage location

All ros2bags should be saved to the SSD here:

```bash
/mnt/ssd/rover_data/ros2bags
```


1. Open a terminal.

2. Source ROS 2 if needed.

3. Start recording:

ros2 bag record -o /mnt/ssd/rover_data/ros2bags/run_$(date +%F_%H-%M-%S) --all

4. Run the simulation or demo.

5. When done, press Ctrl+C.

The bag will be saved on the SSD in /mnt/ssd/rover_data/ros2bags.

6. Inspect it later with:

ros2 bag info /mnt/ssd/rover_data/ros2bags/run_YYYY-MM-DD_HH-MM-SS

7. Replay it later with:

ros2 bag play /mnt/ssd/rover_data/ros2bags/run_YYYY-MM-DD_HH-MM-SS

8. For important captures, --all is fine.

For routine testing, consider recording only the topics you actually need.

Example:

ros2 bag record -o /mnt/ssd/rover_data/ros2bags/test_run /camera/image_raw /scan /tf

Only use topic-specific recording if the team already knows which topics matter for that test.
