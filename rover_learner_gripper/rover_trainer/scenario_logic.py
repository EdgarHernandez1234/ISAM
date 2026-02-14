import time

class ScenarioManager:
    def __init__(self):
        # State Machine for Sticky Mode
        self.sticky_state = "IDLE" 
        self.sticky_timer = 0.0
        self.start_time = 0.0

    def evaluate(self, packet, yolo_results) -> str:
        """
        Returns the ACTION string (e.g., 'SCOOP', 'RETREAT', 'WIGGLE')
        packet: SensorPacket from hardware_manager
        yolo_results: The detection object
        """
        
        # 1. PARSE SENSORS
        dist = packet.min_dist
        darkness = packet.brightness # 0 (Pitch Black) to 255 (White)
        
        # 2. PARSE YOLO (The Human Filter)
        obj_class = "none"
        is_human = False
        if yolo_results:
            # Assuming getting first box
            if len(yolo_results[0].boxes) > 0:
                cls_id = int(yolo_results[0].boxes[0].cls[0])
                obj_class = yolo_results[0].names[cls_id]
                if obj_class == "person":
                    is_human = True

        # --- SCENARIO 3: ADAPTIVE (The "Panic" Mode) ---
        # Trigger: Camera covered (Dark) OR Lidar Timeout (dist is None for too long)
        if darkness < 30.0: # Very dark
            return "ADAPTIVE_PANIC" # Cower/Stop

        # --- SCENARIO 1: UNBREAKABLE (Safety Defense) ---
        # Trigger: Close object that is NOT a human
        if dist is not None and dist < 0.40:
            if is_human:
                print(f"[Logic] Ignoring Human at {dist:.2f}m")
                return "STOP" # Just pause for human, don't retreat
            else:
                return "RETREAT" # Run away from rocks/walls

        # --- SCENARIO 2: STICKY (The Sampling Cycle) ---
        # Trigger: We see "clean" sand/regolith and it's safe
        if obj_class in ["clean", "sand", "regolith", "dirt"]:
            return self._run_sticky_routine()
        
        # Default
        self.sticky_state = "IDLE"
        return "SEARCH"

    def _run_sticky_routine(self) -> str:
        """Handles the multi-step scoop process."""
        now = time.time()
        
        if self.sticky_state == "IDLE":
            self.sticky_state = "SCOOP"
            self.start_time = now
            return "SCOOP"
            
        elapsed = now - self.start_time
        
        if self.sticky_state == "SCOOP":
            if elapsed > 3.0: # Scooping takes 3s
                self.sticky_state = "DUMP"
                return "DUMP"
            return "SCOOP"
            
        elif self.sticky_state == "DUMP":
            if elapsed > 6.0: # Dumping takes 3s
                self.sticky_state = "WIGGLE"
                return "WIGGLE"
            return "DUMP"
            
        elif self.sticky_state == "WIGGLE":
            if elapsed > 9.0: # Wiggling takes 3s
                self.sticky_state = "RETREAT"
                return "RETREAT"
            return "WIGGLE"
            
        elif self.sticky_state == "RETREAT":
            if elapsed > 12.0:
                self.sticky_state = "IDLE" # Done
                return "HOME"
            return "RETREAT"
            
        return "SEARCH"