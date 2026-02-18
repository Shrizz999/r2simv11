#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, PointCloud2
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import math
import sensor_msgs_py.point_cloud2 as pc2

# --- THE EXPANDED STATE MACHINE ---
STATE_SEARCH_SPEARHEAD      = 0   
STATE_ALIGN_SPEARHEAD       = 1   
STATE_APPROACH_SPEARHEAD    = 2   
STATE_DELAY_SPEARHEAD       = 3   
STATE_TURN_180              = 4   
STATE_MOVE_2000MM           = 5   
STATE_TURN_RIGHT_90         = 6   
STATE_SEARCH_200            = 7   
STATE_ALIGN_200             = 8   
STATE_APPROACH_200          = 9   
STATE_PARALLEL_200          = 10   
STATE_CLIMB_200             = 11   
STATE_SEARCH_400            = 12   
STATE_ALIGN_400             = 13
STATE_APPROACH_400          = 14
STATE_PARALLEL_400          = 15  
STATE_CLIMB_400             = 16  
STATE_SEARCH_600            = 17
STATE_ALIGN_600             = 18
STATE_APPROACH_600          = 19
STATE_PARALLEL_600          = 20  
STATE_CLIMB_600             = 21  
STATE_DESCEND_400           = 22  
STATE_TURN_LEFT             = 23  
STATE_DESCEND_200           = 24  
STATE_DESCEND_GROUND        = 25  
STATE_TURN_RIGHT_GROUND     = 26  
STATE_MOVE_1000MM           = 27  # RE-ADDED: Slow, blind 1000mm push
STATE_SEARCH_RAMP           = 28  # REVERTED: Discrete, stable ramp sequence
STATE_ALIGN_RAMP            = 29  
STATE_APPROACH_RAMP         = 30  
STATE_PARALLEL_RAMP         = 31  
STATE_CLIMB_RAMP            = 32  
STATE_TURN_RIGHT_POST_RAMP  = 33  
STATE_SEARCH_TTT            = 34  
STATE_ALIGN_TTT             = 35  
STATE_APPROACH_TTT          = 36  
STATE_FINISHED              = 37

class VisionTracker(Node):
    def __init__(self):
        super().__init__('vision_tracker')
        self.get_logger().info('--- R2KRISHNA: SLOW BLIND PUSH & STABLE RAMP CLIMB ACTIVE ---')

        # --- TUNING ---
        self.SEARCH_SPEED = 0.30     
        self.APPROACH_FWD = 0.30     
        self.CLOSE_STOP_Y = 350    
        
        # --- CRAWLER SPEEDS ---
        self.CLIMB_SPEED = 0.64         
        self.ANTI_FLIP_SPEED = 0.32     
        self.DESCEND_SPEED = 0.24    
        
        # --- FRONT CAMERA VARIABLES ---
        self.state = STATE_SEARCH_SPEARHEAD
        self.br = CvBridge()
        self.block_x = 320
        self.block_y = 240
        self.bottom_y = 240         
        self.target_visible = False
        self.lock_count = 0
        self.target_color = 'BROWN' 
        
        # --- BELLY CAMERA VARIABLES ---
        self.belly_target_color = 'NONE'
        self.belly_target_visible = False

        self.pitch = 0.0
        self.yaw = 0.0
        self.target_yaw = 0.0
        self.delay_start_time = 0.0 
        
        # Climb detection variables
        self.is_pitched = False 
        self.climb_ticks = 0
        self.flat_count = 0
        
        # 3D LiDAR Variables
        self.front_distance = 9.99  
        self.skew_error = 0.0       

        # --- SUBSCRIBERS & PUBLISHERS ---
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(Image, '/belly_camera/image_raw', self.belly_image_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(PointCloud2, '/scan_3d/points', self.lidar_callback, 10)

        self.pub_vel = self.create_publisher(Twist, '/cmd_vel', 10)        
        self.pub_front = self.create_publisher(Twist, '/cmd_vel_front', 10)

        self.create_timer(0.05, self.control_loop)
        cv2.namedWindow("Debug View", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Belly View", cv2.WINDOW_NORMAL)

    def lidar_callback(self, msg):
        try:
            pts = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            x_left = []
            x_right = []
            min_x = 9.99

            for p in pts:
                x, y, z = p[0], p[1], p[2]

                if 0.25 < x < 1.5 and z > -0.15:
                    if abs(y) < 0.20:
                        if x < min_x: min_x = x

                    if 0.10 < y < 0.30:
                        x_left.append(x)
                    elif -0.30 < y < -0.10:
                        x_right.append(x)
            
            self.front_distance = min_x

            if x_left and x_right:
                dist_l = sum(x_left) / len(x_left)
                dist_r = sum(x_right) / len(x_right)
                self.skew_error = dist_l - dist_r
            else:
                self.skew_error = 0.0
        except Exception:
            pass

    def get_hsv_range(self, color_name):
        # We use HSV instead of raw RGB because it is far more stable against shadows and bright reflections.
        if color_name == 'BROWN':        
            return np.array([0, 50, 20]), np.array([30, 255, 200])
        elif color_name == 'DARK_GREEN': 
            return np.array([38, 100, 30]), np.array([54, 255, 180])
        elif color_name == 'MID_GREEN':  
            return np.array([55, 100, 50]), np.array([75, 255, 200])
        elif color_name == 'LIGHT_GREEN':
            return np.array([20, 30, 40]), np.array([55, 255, 255])
        elif color_name == 'WHITE_GREY':       
            # Strict low saturation to only find the ramp and completely ignore the red floor
            return np.array([0, 0, 50]), np.array([180, 40, 255])
        elif color_name == 'YELLOW':     
            return np.array([20, 100, 100]), np.array([40, 255, 255])
        else:
            return np.array([0, 0, 0]), np.array([0, 0, 0])

    def imu_callback(self, msg):
        q = msg.orientation
        sinp = 2 * (q.w * q.y - q.z * q.x)
        self.pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi/2, sinp)
        
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def image_callback(self, msg):
        if self.target_color == 'NONE': return

        try:
            frame = self.br.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            lower, upper = self.get_hsv_range(self.target_color)
            mask = cv2.inRange(hsv, lower, upper)
            
            height, width = mask.shape
            
            # Standard safety mask (always ignores the robot's own bumper)
            mask[int(height*0.95):height, :] = 0 
            
            # UPPER HALF ROI FOR LONG DISTANCE RAMP TRACKING
            if self.target_color == 'WHITE_GREY':
                mask[int(height*0.5):height, :] = 0

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]
                if valid_contours:
                    c = max(valid_contours, key=cv2.contourArea)
                    
                    x, y, w, h = cv2.boundingRect(c)
                    self.bottom_y = y + h  
                    
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        self.block_x = int(M["m10"] / M["m00"])
                        self.block_y = int(M["m01"] / M["m00"])
                        self.target_visible = True
                        
                        cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                        cv2.circle(frame, (self.block_x, self.block_y), 5, (0, 0, 255), -1)
                        cv2.line(frame, (x, self.bottom_y), (x + w, self.bottom_y), (0, 255, 255), 3)
                        
                        state_tag = "SEARCHING" if self.state == STATE_SEARCH_RAMP else "ALIGNING"
                        cv2.putText(frame, f"TARGET: {self.target_color} | {state_tag}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else: self.target_visible = False
            else: self.target_visible = False
            
            cv2.imshow("Debug View", frame)
            cv2.waitKey(1)
        except Exception: pass

    def belly_image_callback(self, msg):
        if self.belly_target_color == 'NONE': 
            self.belly_target_visible = False
            return
            
        try:
            frame = self.br.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            lower, upper = self.get_hsv_range(self.belly_target_color)
            mask = cv2.inRange(hsv, lower, upper)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
                if valid_contours:
                    c = max(valid_contours, key=cv2.contourArea)
                    self.belly_target_visible = True
                    cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                    cv2.putText(frame, f"BELLY LOCK: {self.belly_target_color}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else: 
                    self.belly_target_visible = False
            else: 
                self.belly_target_visible = False
                
            cv2.imshow("Belly View", frame)
            cv2.waitKey(1)
        except Exception: pass

    def drive(self, fwd, rot, front_too=False):
        cmd = Twist()
        cmd.linear.x = float(fwd)
        cmd.angular.z = float(rot)
        self.pub_vel.publish(cmd)
        
        if front_too or (abs(fwd) < 0.01 and abs(rot) > 0.01):
            self.pub_front.publish(cmd)
        else:
            self.pub_front.publish(Twist())

    def reset_climb_vars(self):
        self.climb_ticks = 0
        self.flat_count = 0
        self.is_pitched = False

    def check_pitch_maneuver(self, next_state_id, log_msg):
        self.climb_ticks += 1
        if abs(self.pitch) > 0.12: self.is_pitched = True
        
        if self.climb_ticks % 20 == 0:
            self.get_logger().info(f"Maneuvering... Pitch: {self.pitch:.3f} | Pitched?: {self.is_pitched}")
            
        if self.is_pitched:
            if abs(self.pitch) < 0.06: 
                self.flat_count += 1
            else:
                self.flat_count = 0
                
            if self.flat_count > 1:
                self.drive(0.0, 0.0)
                self.get_logger().info(log_msg)
                self.state = next_state_id
                self.reset_climb_vars()

    def check_belly_descent_maneuver(self, next_state_id, log_msg, target_color):
        self.belly_target_color = target_color
        self.climb_ticks += 1
        
        if abs(self.pitch) > 0.12:  
            self.is_pitched = True
            
        if self.climb_ticks % 20 == 0:
            self.get_logger().info(f"Descending... Pitch: {self.pitch:.3f} | Belly sees {target_color}?: {self.belly_target_visible}")
            
        if self.is_pitched:
            if abs(self.pitch) < 0.06:
                self.flat_count += 1
            else:
                self.flat_count = 0
                
            if (self.flat_count > 0 and self.belly_target_visible) or self.flat_count > 4:
                self.drive(0.0, 0.0)
                self.get_logger().info(log_msg)
                self.belly_target_color = 'NONE' 
                self.state = next_state_id
                self.reset_climb_vars()

    def control_loop(self):
        if self.state == STATE_FINISHED:
            self.drive(0, 0)
            return

        # ==========================================
        # 0-3. SPEARHEAD RACK
        # ==========================================
        if self.state == STATE_SEARCH_SPEARHEAD:
            if self.target_visible:
                self.lock_count += 1
                if self.lock_count > 5: self.state = STATE_ALIGN_SPEARHEAD
            else:
                self.lock_count = 0
                self.drive(0.0, self.SEARCH_SPEED) 

        elif self.state == STATE_ALIGN_SPEARHEAD:
            if not self.target_visible: self.state = STATE_SEARCH_SPEARHEAD; return
            error = self.block_x - 320
            if abs(error) > 10: self.drive(0.0, error * 0.005) 
            else:
                self.get_logger().info("Spearhead Locked! Approaching.")
                self.state = STATE_APPROACH_SPEARHEAD
                self.drive(0.0, 0.0)

        elif self.state == STATE_APPROACH_SPEARHEAD:
            turn = (self.block_x - 320) * 0.003 if self.target_visible else 0.0
            self.drive(self.APPROACH_FWD, turn)
            
            if self.bottom_y > self.CLOSE_STOP_Y or not self.target_visible:
                self.drive(0.0, 0.0)
                self.delay_start_time = time.time()
                self.get_logger().info("Reached Spearhead (300mm). Waiting 2 seconds...")
                self.state = STATE_DELAY_SPEARHEAD
                
        elif self.state == STATE_DELAY_SPEARHEAD:
            self.drive(0.0, 0.0) 
            if (time.time() - self.delay_start_time) >= 2.0:
                self.target_yaw = self.yaw + math.pi
                while self.target_yaw <= -math.pi: self.target_yaw += 2 * math.pi
                while self.target_yaw > math.pi: self.target_yaw -= 2 * math.pi
                
                self.get_logger().info("Wait complete! Executing 180 Turn.")
                self.state = STATE_TURN_180

        # ==========================================
        # 4-6. THE 2000mm WAYPOINT BYPASS
        # ==========================================
        elif self.state == STATE_TURN_180:
            yaw_error = self.target_yaw - self.yaw
            while yaw_error <= -math.pi: yaw_error += 2 * math.pi
            while yaw_error > math.pi: yaw_error -= 2 * math.pi
            
            if abs(yaw_error) > 0.15: 
                turn_speed = max(min(-yaw_error * 1.5, 0.6), -0.6)
                if 0 < turn_speed < 0.15: turn_speed = 0.15
                if -0.15 < turn_speed < 0: turn_speed = -0.15
                self.drive(0.0, turn_speed)
            else:
                self.drive(0.0, 0.0)
                self.delay_start_time = time.time() 
                self.get_logger().info("Turn complete. Driving 2000mm forward at 0.48 m/s.")
                self.state = STATE_MOVE_2000MM
                
        elif self.state == STATE_MOVE_2000MM:
            if (time.time() - self.delay_start_time) < 4.16:
                self.drive(0.48, 0.0)
            else:
                self.drive(0.0, 0.0)
                self.target_yaw = self.yaw - (math.pi / 2.0)
                while self.target_yaw <= -math.pi: self.target_yaw += 2 * math.pi
                while self.target_yaw > math.pi: self.target_yaw -= 2 * math.pi
                
                self.get_logger().info("2000mm Reached. Turning RIGHT to face Stairs.")
                self.state = STATE_TURN_RIGHT_90

        elif self.state == STATE_TURN_RIGHT_90:
            yaw_error = self.target_yaw - self.yaw
            while yaw_error <= -math.pi: yaw_error += 2 * math.pi
            while yaw_error > math.pi: yaw_error -= 2 * math.pi
            
            if abs(yaw_error) > 0.15: 
                turn_speed = max(min(-yaw_error * 1.5, 0.6), -0.6)
                if 0 < turn_speed < 0.15: turn_speed = 0.15
                if -0.15 < turn_speed < 0: turn_speed = -0.15
                self.drive(0.0, turn_speed)
            else:
                self.drive(0.0, 0.0)
                self.get_logger().info("Facing Stairs! Hunting Sandwiched 200mm Stair.")
                self.target_color = 'DARK_GREEN'
                self.state = STATE_SEARCH_200

        # ==========================================
        # 7-11. CLIMB 200mm STAIR
        # ==========================================
        elif self.state == STATE_SEARCH_200:
            if self.target_visible:
                self.lock_count += 1
                if self.lock_count > 5: self.state = STATE_ALIGN_200
            else:
                self.lock_count = 0
                self.drive(0.0, self.SEARCH_SPEED) 

        elif self.state == STATE_ALIGN_200:
            if not self.target_visible: self.state = STATE_SEARCH_200; return
            error = self.block_x - 320
            if abs(error) > 10: self.drive(0.0, error * 0.005) 
            else:
                self.state = STATE_APPROACH_200
                self.drive(0.0, 0.0)

        elif self.state == STATE_APPROACH_200:
            turn = (self.block_x - 320) * 0.003 if self.target_visible else 0.0
            self.drive(self.APPROACH_FWD, turn)
            
            if self.front_distance <= 0.45:
                self.drive(0.0, 0.0)
                self.get_logger().info("300mm Standoff Reached. Auto-Centering in Sandwich Gap...")
                self.state = STATE_PARALLEL_200

        elif self.state == STATE_PARALLEL_200:
            if abs(self.skew_error) > 0.02 and self.front_distance < 0.6:
                turn_speed = max(min(self.skew_error * 2.0, 0.3), -0.3)
                if 0 < turn_speed < 0.15: turn_speed = 0.15
                if -0.15 < turn_speed < 0: turn_speed = -0.15
                self.drive(0.0, turn_speed)
            else:
                self.drive(0.0, 0.0)
                self.get_logger().info("PARALLEL ALIGNMENT PERFECT. Climbing 200mm!")
                self.reset_climb_vars()
                self.state = STATE_CLIMB_200

        elif self.state == STATE_CLIMB_200:
            safe_speed = self.ANTI_FLIP_SPEED if abs(self.pitch) > 0.20 else self.CLIMB_SPEED
            self.drive(safe_speed, 0.0, front_too=True)
            self.check_pitch_maneuver(STATE_SEARCH_400, "Climbed 200mm! Hunting 400mm Stair.")
            if self.state == STATE_SEARCH_400: self.target_color = 'MID_GREEN'

        # ==========================================
        # 12-16. CLIMB 400mm STAIR
        # ==========================================
        elif self.state == STATE_SEARCH_400:
            if self.target_visible:
                self.lock_count += 1
                if self.lock_count > 5: self.state = STATE_ALIGN_400
            else:
                self.lock_count = 0
                self.drive(0.20, 0.0)

        elif self.state == STATE_ALIGN_400:
            if not self.target_visible: self.state = STATE_SEARCH_400; return
            error = self.block_x - 320
            if abs(error) > 10: self.drive(0.0, error * 0.005)
            else:
                self.state = STATE_APPROACH_400
                self.drive(0.0, 0.0)

        elif self.state == STATE_APPROACH_400:
            turn = (self.block_x - 320) * 0.003 if self.target_visible else 0.0
            self.drive(self.APPROACH_FWD, turn)
            
            if self.front_distance <= 0.45:
                self.drive(0.0, 0.0)
                self.get_logger().info("300mm Standoff Reached. Aligning Parallel...")
                self.state = STATE_PARALLEL_400

        elif self.state == STATE_PARALLEL_400:
            if abs(self.skew_error) > 0.02 and self.front_distance < 0.6:
                turn_speed = max(min(self.skew_error * 2.0, 0.3), -0.3)
                if 0 < turn_speed < 0.15: turn_speed = 0.15
                if -0.15 < turn_speed < 0: turn_speed = -0.15
                self.drive(0.0, turn_speed)
            else:
                self.drive(0.0, 0.0)
                self.get_logger().info("PARALLEL ALIGNMENT PERFECT. Climbing 400mm!")
                self.reset_climb_vars()
                self.state = STATE_CLIMB_400

        elif self.state == STATE_CLIMB_400:
            safe_speed = self.ANTI_FLIP_SPEED if abs(self.pitch) > 0.20 else self.CLIMB_SPEED
            self.drive(safe_speed, 0.0, front_too=True)
            self.check_pitch_maneuver(STATE_SEARCH_600, "Climbed 400mm! Hunting 600mm Stair.")
            if self.state == STATE_SEARCH_600: self.target_color = 'LIGHT_GREEN'

        # ==========================================
        # 17-21. CLIMB 600mm STAIR
        # ==========================================
        elif self.state == STATE_SEARCH_600:
            if self.target_visible:
                self.lock_count += 1
                if self.lock_count > 5: self.state = STATE_ALIGN_600
            else:
                self.lock_count = 0
                self.drive(0.20, 0.0)

        elif self.state == STATE_ALIGN_600:
            if not self.target_visible: self.state = STATE_SEARCH_600; return
            error = self.block_x - 320
            if abs(error) > 10: self.drive(0.0, error * 0.005)
            else:
                self.state = STATE_APPROACH_600
                self.drive(0.0, 0.0)

        elif self.state == STATE_APPROACH_600:
            turn = (self.block_x - 320) * 0.003 if self.target_visible else 0.0
            self.drive(self.APPROACH_FWD, turn)
            
            if self.front_distance <= 0.45:
                self.drive(0.0, 0.0)
                self.get_logger().info("300mm Standoff Reached. Aligning Parallel...")
                self.state = STATE_PARALLEL_600

        elif self.state == STATE_PARALLEL_600:
            if abs(self.skew_error) > 0.02 and self.front_distance < 0.6:
                turn_speed = max(min(self.skew_error * 2.0, 0.3), -0.3)
                if 0 < turn_speed < 0.15: turn_speed = 0.15
                if -0.15 < turn_speed < 0: turn_speed = -0.15
                self.drive(0.0, turn_speed)
            else:
                self.drive(0.0, 0.0)
                self.get_logger().info("PARALLEL ALIGNMENT PERFECT. Climbing 600mm!")
                self.reset_climb_vars()
                self.state = STATE_CLIMB_600

        elif self.state == STATE_CLIMB_600:
            safe_speed = self.ANTI_FLIP_SPEED if abs(self.pitch) > 0.20 else self.CLIMB_SPEED
            self.drive(safe_speed, 0.0, front_too=True)
            self.check_pitch_maneuver(STATE_DESCEND_400, "Climbed 600mm! Descending to 400mm.")

        # ==========================================
        # 22-26. DESCEND SEQUENCE TO GROUND
        # ==========================================
        elif self.state == STATE_DESCEND_400:
            self.drive(self.DESCEND_SPEED, 0.0, front_too=True)
            self.check_belly_descent_maneuver(STATE_TURN_LEFT, "Belly Confirmed 400mm Landing! Turning LEFT.", 'MID_GREEN')
            
            if self.state == STATE_TURN_LEFT:
                self.target_yaw = self.yaw + math.radians(90)
                while self.target_yaw <= -math.pi: self.target_yaw += 2 * math.pi
                while self.target_yaw > math.pi: self.target_yaw -= 2 * math.pi

        elif self.state == STATE_TURN_LEFT:
            yaw_error = self.target_yaw - self.yaw
            while yaw_error <= -math.pi: yaw_error += 2 * math.pi
            while yaw_error > math.pi: yaw_error -= 2 * math.pi
            
            if abs(yaw_error) > 0.15: 
                turn_speed = max(min(-yaw_error * 1.5, 0.6), -0.6) 
                if 0 < turn_speed < 0.15: turn_speed = 0.15
                if -0.15 < turn_speed < 0: turn_speed = -0.15
                self.drive(0.0, turn_speed)
            else:
                self.drive(0.0, 0.0)
                self.get_logger().info("Turn complete. Descending to 200mm.")
                self.reset_climb_vars()
                self.state = STATE_DESCEND_200

        elif self.state == STATE_DESCEND_200:
            self.drive(self.DESCEND_SPEED, 0.0, front_too=True)
            self.check_belly_descent_maneuver(STATE_DESCEND_GROUND, "Belly Confirmed 200mm Landing! Continuing descent to Ground.", 'DARK_GREEN')

        elif self.state == STATE_DESCEND_GROUND:
            self.drive(self.DESCEND_SPEED, 0.0, front_too=True)
            self.check_pitch_maneuver(STATE_TURN_RIGHT_GROUND, "Landed on ground! Turning RIGHT.")
            
            if self.state == STATE_TURN_RIGHT_GROUND:
                self.target_yaw = self.yaw - math.radians(90) 
                while self.target_yaw <= -math.pi: self.target_yaw += 2 * math.pi
                while self.target_yaw > math.pi: self.target_yaw -= 2 * math.pi

        elif self.state == STATE_TURN_RIGHT_GROUND:
            yaw_error = self.target_yaw - self.yaw
            while yaw_error <= -math.pi: yaw_error += 2 * math.pi
            while yaw_error > math.pi: yaw_error -= 2 * math.pi
            
            if abs(yaw_error) > 0.15: 
                turn_speed = max(min(-yaw_error * 1.5, 0.6), -0.6) 
                if 0 < turn_speed < 0.15: turn_speed = 0.15
                if -0.15 < turn_speed < 0: turn_speed = -0.15
                self.drive(0.0, turn_speed)
            else:
                self.drive(0.0, 0.0)
                self.get_logger().info("Ground Turn complete. Moving 1000mm STRAIGHT and SLOWLY.")
                self.delay_start_time = time.time()
                
                # Turn OFF camera tracking during the blind move to prevent accidental locks
                self.target_color = 'NONE' 
                self.state = STATE_MOVE_1000MM

        # ==========================================
        # 27. THE 1000mm SLOW, BLIND PUSH
        # ==========================================
        elif self.state == STATE_MOVE_1000MM:
            # Move slowly at 0.25 m/s for exactly 4.0 seconds (= 1000mm)
            if (time.time() - self.delay_start_time) < 4.0:
                self.drive(0.25, 0.0) # Strictly straight, ZERO turning
            else:
                self.drive(0.0, 0.0)
                self.get_logger().info("1000mm move complete. Detecting Ramp to align and climb.")
                self.reset_climb_vars()
                self.target_color = 'WHITE_GREY'
                self.state = STATE_SEARCH_RAMP

        # ==========================================
        # 28-32. THE RAMP (STABLE SEQUENCE)
        # ==========================================
        elif self.state == STATE_SEARCH_RAMP:
            if self.target_visible:
                self.lock_count += 1
                if self.lock_count > 5: 
                    self.state = STATE_ALIGN_RAMP
            else:
                self.lock_count = 0
                self.drive(0.0, self.SEARCH_SPEED)

        elif self.state == STATE_ALIGN_RAMP:
            if not self.target_visible: 
                self.state = STATE_SEARCH_RAMP
                return
            error = self.block_x - 320
            if abs(error) > 10: 
                self.drive(0.0, error * 0.005)
            else:
                self.state = STATE_APPROACH_RAMP
                self.drive(0.0, 0.0)

        elif self.state == STATE_APPROACH_RAMP:
            turn = (self.block_x - 320) * 0.003 if self.target_visible else 0.0
            self.drive(self.APPROACH_FWD, turn)
            
            if self.front_distance <= 0.45:
                self.drive(0.0, 0.0)
                self.get_logger().info("300mm Standoff Reached. Aligning Parallel...")
                self.state = STATE_PARALLEL_RAMP

        elif self.state == STATE_PARALLEL_RAMP:
            if abs(self.skew_error) > 0.02 and self.front_distance < 0.6:
                turn_speed = max(min(self.skew_error * 2.0, 0.3), -0.3)
                if 0 < turn_speed < 0.15: turn_speed = 0.15
                if -0.15 < turn_speed < 0: turn_speed = -0.15
                self.drive(0.0, turn_speed)
            else:
                self.drive(0.0, 0.0)
                self.get_logger().info("PARALLEL ALIGNMENT PERFECT. Climbing Ramp!")
                self.reset_climb_vars()
                self.state = STATE_CLIMB_RAMP

        elif self.state == STATE_CLIMB_RAMP:
            safe_speed = self.ANTI_FLIP_SPEED if abs(self.pitch) > 0.20 else self.CLIMB_SPEED
            self.drive(safe_speed, 0.0, front_too=True)
            self.check_pitch_maneuver(STATE_TURN_RIGHT_POST_RAMP, "Climbed Ramp! Turning RIGHT.")
            if self.state == STATE_TURN_RIGHT_POST_RAMP:
                self.target_yaw = self.yaw - math.radians(80) 
                while self.target_yaw <= -math.pi: self.target_yaw += 2 * math.pi
                while self.target_yaw > math.pi: self.target_yaw -= 2 * math.pi

        # ==========================================
        # 33-37. TIC-TAC-TOE BOARD
        # ==========================================
        elif self.state == STATE_TURN_RIGHT_POST_RAMP:
            yaw_error = self.target_yaw - self.yaw
            while yaw_error <= -math.pi: yaw_error += 2 * math.pi
            while yaw_error > math.pi: yaw_error -= 2 * math.pi
            
            if abs(yaw_error) > 0.15: 
                turn_speed = max(min(-yaw_error * 1.5, 0.6), -0.6)
                if 0 < turn_speed < 0.15: turn_speed = 0.15
                if -0.15 < turn_speed < 0: turn_speed = -0.15
                self.drive(0.0, turn_speed)
            else:
                self.drive(0.0, 0.0)
                self.get_logger().info("Turn complete. Hunting Tic-Tac-Toe.")
                self.target_color = 'YELLOW'
                self.state = STATE_SEARCH_TTT

        elif self.state == STATE_SEARCH_TTT:
            if self.target_visible:
                self.lock_count += 1
                if self.lock_count > 5: self.state = STATE_ALIGN_TTT
            else:
                self.lock_count = 0
                self.drive(0.0, self.SEARCH_SPEED)

        elif self.state == STATE_ALIGN_TTT:
            if not self.target_visible: self.state = STATE_SEARCH_TTT; return
            error = self.block_x - 320
            if abs(error) > 20: self.drive(0.0, error * 0.005)
            else:
                self.state = STATE_APPROACH_TTT
                self.drive(0.0, 0.0)

        elif self.state == STATE_APPROACH_TTT:
            turn = (self.block_x - 320) * 0.003 if self.target_visible else 0.0
            self.drive(self.APPROACH_FWD, turn)
            
            if self.bottom_y > 360 or not self.target_visible:
                self.drive(0.0, 0.0)
                self.get_logger().info("ARRIVED AT TIC-TAC-TOE! SEQUENCE COMPLETE.")
                self.state = STATE_FINISHED

if __name__ == '__main__':
    rclpy.init()
    node = VisionTracker()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()