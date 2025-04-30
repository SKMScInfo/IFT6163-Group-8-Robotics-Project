#!/usr/bin/env python3

import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from nav2_msgs.action import NavigateToPose
from nav2_simple_commander.robot_navigator import BasicNavigator
import numpy as np
import math
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration

class WallAwareExplorer(Node):
    def __init__(self):
        super().__init__('wall_aware_explorer')
        self.get_logger().info("Starting Wall-Aware Explorer Node")
        
        # Initialize navigator with configurable timeout
        self.get_logger().info("Waiting for Nav2 to become active (timeout: 60s)...")
        self.navigator = BasicNavigator()
        try:
            # Use a timeout to avoid hanging indefinitely
            self.navigator.waitUntilNav2Active(navigator_ready_timeout=60.0)
            self.get_logger().info("Nav2 is active and ready!")
        except Exception as e:
            self.get_logger().error(f"Error initializing Nav2: {str(e)}")
            self.get_logger().warn("Will continue anyway and try to connect later")
            
        # Create subscription to map topic
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        
        # Direct velocity command publisher for manual recovery behaviors
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        # Tracking variables
        self.map_data = None
        self.exploring = False
        self.visited_frontiers = []
        self.failed_goals = []
        self.goal_start_time = None
        self.current_goal = None
        self.recovery_in_progress = False
        self.last_recovery_time = 0
        self.force_new_goal_after_recovery = False
        
        # Parameters
        self.frontier_distance_threshold = 0.8  # Meters between frontiers
        self.goal_timeout = 90.0  # Increased timeout from 30s to 90s
        self.map_update_frequency = 2.0  # Seconds between map processing attempts
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        self.min_frontier_wall_distance = 5  # Grid cells from obstacles
        self.recovery_cooldown = 15.0  # Reduced cooldown from 30s to 15s
        self.recovery_strategies = ['rotate', 'wiggle', 'back_up']  # Different recovery behaviors
        self.current_recovery_strategy = 0  # Index to track which recovery to try next
        
        # Multi-tiered frontiers organization by quality
        self.frontier_tiers = {
            'excellent': [],  # Very far from walls
            'good': [],       # Reasonably far from walls
            'acceptable': [], # Further from walls than minimum
            'last_resort': [] # Meeting minimum requirements
        }
        
        # Create timer for continuous exploration
        self.explore_timer = self.create_timer(
            1.0/self.map_update_frequency, 
            self.exploration_loop
        )
        
        self.get_logger().info("Wall-aware explorer initialized and ready")
        self.get_logger().info(f"Using goal timeout of {self.goal_timeout} seconds")

    def map_callback(self, msg):
        """Store map data when received"""
        self.map_data = msg
        if not self.exploring and self.map_data is not None:
            self.exploring = True
            self.get_logger().info("Map received. Ready to start exploration.")

    def exploration_loop(self):
        """Main exploration logic, runs periodically"""
        if not self.exploring or self.map_data is None:
            return
            
        # Skip processing if we're in recovery mode
        if self.recovery_in_progress:
            return
            
        # Check if we're in recovery cooldown period
        time_since_recovery = time.time() - self.last_recovery_time
        if self.last_recovery_time > 0 and time_since_recovery < self.recovery_cooldown:
            # Only log occasionally to avoid spamming
            if int(time_since_recovery) % 5 == 0:
                remaining = self.recovery_cooldown - time_since_recovery
                self.get_logger().info(f"In recovery cooldown. {remaining:.1f}s remaining.")
            return
            
        # Check if we need to send a new goal after recovery cooldown
        force_new_goal = False
        if self.last_recovery_time > 0 and time_since_recovery >= self.recovery_cooldown and time_since_recovery < self.recovery_cooldown + 1.0:
            self.get_logger().info("Recovery cooldown complete. Actively seeking new frontier.")
            self.last_recovery_time = 0  # Reset recovery time
            self.current_goal = None  # Clear any previous goal
            force_new_goal = True  # Force setting a new goal
            
        # Check current navigation status
        if self.current_goal is not None and self.goal_start_time is not None:
            # Check for timeout on current goal
            elapsed = time.time() - self.goal_start_time
            if elapsed > self.goal_timeout:
                self.get_logger().warn(f"Goal timeout reached after {elapsed:.1f}s! Initiating recovery.")
                self.cancel_current_goal()
                self.execute_recovery()
                return
        
        # Check if navigator is available for a new goal or if we're forcing a new goal
        if force_new_goal or self.navigator.isTaskComplete():
            # Get result of previous navigation, if any
            if self.current_goal is not None and not force_new_goal:
                result = self.navigator.getResult()
                
                if result == "SUCCEEDED":
                    self.get_logger().info("Goal succeeded!")
                    self.consecutive_failures = 0
                elif result in ["FAILED", "CANCELED"]:
                    self.consecutive_failures += 1
                    self.get_logger().warn(f"Navigation {result}. Consecutive failures: {self.consecutive_failures}")
                    
                    # Add this to failed goals to avoid retrying
                    if self.current_goal is not None:
                        self.failed_goals.append((self.current_goal.pose.position.x, 
                                                 self.current_goal.pose.position.y))
                    
                    # If we've had too many consecutive failures, try recovery behaviors
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        self.get_logger().warn(f"Too many consecutive failures. Executing recovery behavior...")
                        self.execute_recovery()
                        return  # Skip finding a new frontier until recovery is complete
                
                self.current_goal = None
            
            # Find and go to next frontier
            self.get_logger().info("Looking for new frontiers to explore")
            self.find_all_frontiers()
            
            # Select the best frontier from any tier that has frontiers
            frontier = self.select_best_frontier()
            
            if frontier is None:
                self.get_logger().info("No more frontiers found. Exploration completed.")
                self.exploring = False
                return
                
            self.get_logger().info(f"New frontier found at: ({frontier[0]:.2f}, {frontier[1]:.2f})")
            self.send_goal(frontier)

    def find_all_frontiers(self):
        """Classify all frontiers into quality tiers based on wall proximity"""
        # Clear previous frontiers
        for tier in self.frontier_tiers.values():
            tier.clear()
            
        # Convert occupancy grid to numpy array
        grid = np.array(self.map_data.data).reshape(
            self.map_data.info.height,
            self.map_data.info.width
        )
        
        # Parameters for wall distance evaluation
        excellent_dist = 12  # cells
        good_dist = 8        # cells
        acceptable_dist = 5  # cells
        minimum_dist = 3     # cells
        
        # Find frontiers: unknown cells (-1) adjacent to free space (0)
        for y in range(1, grid.shape[0]-1):
            for x in range(1, grid.shape[1]-1):
                if grid[y, x] == -1:  # Unknown space
                    neighborhood = grid[y-1:y+2, x-1:x+2]
                    if np.any(neighborhood == 0):  # Adjacent to free space
                        # Calculate wall distance score
                        wall_distance = self.calculate_wall_distance(grid, x, y)
                        
                        # Convert grid coordinates to world coordinates
                        goal_x = x * self.map_data.info.resolution + self.map_data.info.origin.position.x
                        goal_y = y * self.map_data.info.resolution + self.map_data.info.origin.position.y
                        
                        # Skip if too close to previously visited frontiers
                        too_close = self.is_too_close_to_visited((goal_x, goal_y))
                        if too_close:
                            continue
                            
                        # Skip if this is a previously failed goal
                        if self.is_failed_goal((goal_x, goal_y)):
                            continue
                        
                        # Classify frontier based on wall distance
                        if wall_distance >= excellent_dist:
                            self.frontier_tiers['excellent'].append((goal_x, goal_y, wall_distance))
                        elif wall_distance >= good_dist:
                            self.frontier_tiers['good'].append((goal_x, goal_y, wall_distance))
                        elif wall_distance >= acceptable_dist:
                            self.frontier_tiers['acceptable'].append((goal_x, goal_y, wall_distance))
                        elif wall_distance >= minimum_dist:
                            self.frontier_tiers['last_resort'].append((goal_x, goal_y, wall_distance))
                        # Discard frontiers too close to walls
    
    def select_best_frontier(self):
        """Select the best frontier from all tiers, prioritizing higher quality tiers"""
        # Try to select from tiers in order of quality
        for tier_name in ['excellent', 'good', 'acceptable', 'last_resort']:
            tier = self.frontier_tiers[tier_name]
            if tier:
                # Sort by wall distance (highest first)
                tier.sort(key=lambda x: x[2], reverse=True)
                
                # Log how many frontiers we found
                self.get_logger().info(f"Found {len(tier)} {tier_name} frontiers")
                
                # Return the coordinates of the best frontier in this tier
                return (tier[0][0], tier[0][1])
        
        return None  # No frontiers found in any tier
    
    def calculate_wall_distance(self, grid, x, y):
        """Calculate minimum distance to any occupied cell"""
        # Simple implementation: check in expanding squares up to max_dist
        max_dist = 15  # Maximum distance to check
        
        for d in range(1, max_dist + 1):
            # Check square perimeter at distance d
            for i in range(-d, d+1):
                for j in range(-d, d+1):
                    # Only check perimeter cells
                    if abs(i) == d or abs(j) == d:
                        xi, yj = x + i, y + j
                        
                        # Check bounds
                        if 0 <= xi < grid.shape[1] and 0 <= yj < grid.shape[0]:
                            # If cell is occupied, return this distance
                            if grid[yj, xi] == 100:  # 100 means occupied
                                return d
        
        # No walls found within max_dist
        return max_dist
    
    def is_too_close_to_visited(self, point):
        """Check if point is too close to any visited frontier"""
        x, y = point
        for visited_x, visited_y in self.visited_frontiers:
            dist = math.sqrt((x - visited_x)**2 + (y - visited_y)**2)
            if dist < self.frontier_distance_threshold:
                return True
        return False
    
    def is_failed_goal(self, point):
        """Check if point is a previously failed goal"""
        x, y = point
        for failed_x, failed_y in self.failed_goals:
            dist = math.sqrt((x - failed_x)**2 + (y - failed_y)**2)
            if dist < self.frontier_distance_threshold:
                return True
        return False

    def send_goal(self, frontier_point):
        """Send navigation goal to frontier with a timeout"""
        x, y = frontier_point
        
        # Remember this frontier to avoid revisiting
        self.visited_frontiers.append((x, y))
        
        # Reset goal start time
        self.goal_start_time = time.time()
        
        # Prepare the goal
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        
        # Set orientation to a valid quaternion
        goal.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        self.get_logger().info(f"Sending goal to frontier at ({x:.2f}, {y:.2f})")
        
        # Send goal to Nav2
        self.navigator.goToPose(goal)
        
        # Store current goal for timeout tracking
        self.current_goal = goal
    
    def cancel_current_goal(self):
        """Cancel the current navigation goal"""
        if self.current_goal is not None:
            self.get_logger().info("Canceling current goal due to timeout or recovery")
            self.navigator.cancelTask()
            # Give Nav2 a moment to process the cancellation
            time.sleep(0.5)
    
    def execute_recovery(self):
        """Execute recovery behaviors when robot is stuck"""
        self.get_logger().info("Starting recovery behavior")
        self.recovery_in_progress = True
        
        # First cancel any current goal
        self.cancel_current_goal()
        
        # Choose a recovery strategy based on rotation
        strategy = self.recovery_strategies[self.current_recovery_strategy]
        self.get_logger().info(f"Using recovery strategy: {strategy}")
        
        # Rotate to next strategy for next time
        self.current_recovery_strategy = (self.current_recovery_strategy + 1) % len(self.recovery_strategies)
        
        # Execute the selected recovery behavior
        if strategy == 'rotate':
            self.rotate_recovery()
        elif strategy == 'wiggle':
            self.wiggle_recovery()
        elif strategy == 'back_up':
            self.back_up_recovery()
        
        # Mark the time when recovery finished
        self.last_recovery_time = time.time()
        
        # Reset state after recovery
        self.recovery_in_progress = False
        self.consecutive_failures = 0
        self.current_goal = None  # Explicitly clear the current goal
        
        # Clear some failed goals to allow re-trying after recovery
        if len(self.failed_goals) > 5:  # Lowered threshold to clear more aggressively
            # Keep the 3 most recent failures
            self.failed_goals = self.failed_goals[-3:]
            self.get_logger().info("Cleared failed goals to allow re-exploration")
        
        # Schedule timer to force a new goal after cooldown
        self.get_logger().info(f"Recovery complete. Entering cooldown for {self.recovery_cooldown}s.")
        # Set flag to force finding a new frontier after cooldown
        self.force_new_goal_after_recovery = True
    
    def rotate_recovery(self):
        """Recovery behavior: rotate in place to scan surroundings"""
        self.get_logger().info("Recovery: Rotating 360 degrees to scan surroundings")
        
        # Create a velocity command for rotation
        twist = Twist()
        twist.angular.z = 0.5  # Rotate at 0.5 rad/s (about 30 deg/s)
        
        # Rotate for about 12 seconds (full 360 degrees)
        start_time = time.time()
        while time.time() - start_time < 12.0:
            # Publish rotation command
            self.cmd_vel_pub.publish(twist)
            # Sleep a bit to avoid flooding
            time.sleep(0.1)
        
        # Stop rotation
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info("Rotation recovery complete")
    
    def wiggle_recovery(self):
        """Recovery behavior: wiggle back and forth to escape small obstacles"""
        self.get_logger().info("Recovery: Performing wiggle recovery")
        
        twist = Twist()
        
        # First wiggle left
        for _ in range(5):
            twist.angular.z = 0.7
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.3)
            
            twist.angular.z = -0.7
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.3)
        
        # Then wiggle right
        for _ in range(5):
            twist.angular.z = -0.7
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.3)
            
            twist.angular.z = 0.7
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.3)
        
        # Stop all movement
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info("Wiggle recovery complete")
    
    def back_up_recovery(self):
        """Recovery behavior: back up to escape from tight spaces"""
        self.get_logger().info("Recovery: Backing up to escape tight space")
        
        # Create a velocity command for backing up
        twist = Twist()
        twist.linear.x = -0.1  # Slow backup
        
        # Back up for 5 seconds
        start_time = time.time()
        while time.time() - start_time < 5.0:
            # Publish backup command
            self.cmd_vel_pub.publish(twist)
            # Sleep a bit to avoid flooding
            time.sleep(0.1)
        
        # Stop movement
        twist.linear.x = 0.0
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info("Backup recovery complete")

def main(args=None):
    rclpy.init(args=args)
    
    # Create an explorer node
    try:
        node = WallAwareExplorer()
        
        # Print help message
        print("\n=====================================================")
        print("FIXED WALL-AWARE EXPLORER NODE")
        print("This node will help navigate around walls and avoid getting stuck")
        print("Make sure navigation stack is running first!")
        print("Goal timeout: 90s with multiple recovery strategies")
        print("Recovery cooldown: 15s with forced new goal selection")
        print("=====================================================\n")
        
        # Spin the node
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Clean shutdown
        try:
            if 'node' in locals():
                node.get_logger().info("Shutting down explorer node")
                node.destroy_node()
        except:
            pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()
