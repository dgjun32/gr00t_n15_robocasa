"""
Comprehensive test suite for RobotTrajectoryPlanner

This test suite covers:
1. Basic pathfinding functionality
2. Edge cases (no path, start=goal, invalid positions)
3. Performance with different obstacle configurations
4. Correctness of A* algorithm implementation
"""

import unittest
import time
from robot_trajectory_planner import RobotTrajectoryPlanner, Point


class TestRobotTrajectoryPlanner(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.planner = RobotTrajectoryPlanner(10, 10)
    
    def test_empty_workspace_path(self):
        """Test pathfinding in empty workspace."""
        start = Point(0, 0)
        goal = Point(9, 9)
        
        path = self.planner.find_path(start, goal)
        
        self.assertIsNotNone(path)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
        self.assertGreater(len(path), 1)
    
    def test_start_equals_goal(self):
        """Test when start position equals goal position."""
        start = Point(5, 5)
        goal = Point(5, 5)
        
        path = self.planner.find_path(start, goal)
        
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 1)
        self.assertEqual(path[0], start)
    
    def test_invalid_start_position(self):
        """Test with invalid start position."""
        start = Point(-1, 0)  # Out of bounds
        goal = Point(5, 5)
        
        path = self.planner.find_path(start, goal)
        self.assertIsNone(path)
    
    def test_invalid_goal_position(self):
        """Test with invalid goal position."""
        start = Point(0, 0)
        goal = Point(10, 10)  # Out of bounds
        
        path = self.planner.find_path(start, goal)
        self.assertIsNone(path)
    
    def test_obstacle_at_start(self):
        """Test when start position is an obstacle."""
        self.planner.add_obstacle(0, 0)
        start = Point(0, 0)
        goal = Point(5, 5)
        
        path = self.planner.find_path(start, goal)
        self.assertIsNone(path)
    
    def test_obstacle_at_goal(self):
        """Test when goal position is an obstacle."""
        self.planner.add_obstacle(5, 5)
        start = Point(0, 0)
        goal = Point(5, 5)
        
        path = self.planner.find_path(start, goal)
        self.assertIsNone(path)
    
    def test_simple_obstacle_avoidance(self):
        """Test pathfinding with simple obstacles."""
        # Add a wall of obstacles
        for y in range(3, 7):
            self.planner.add_obstacle(5, y)
        
        start = Point(0, 5)
        goal = Point(9, 5)
        
        path = self.planner.find_path(start, goal)
        
        self.assertIsNotNone(path)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
        
        # Verify no path point is an obstacle
        for point in path:
            self.assertNotIn(point, self.planner.obstacles)
    
    def test_no_path_exists(self):
        """Test when no path exists due to obstacles."""
        # Create a wall that completely blocks the path
        for x in range(10):
            self.planner.add_obstacle(x, 5)
        
        start = Point(0, 0)
        goal = Point(9, 9)
        
        path = self.planner.find_path(start, goal)
        self.assertIsNone(path)
    
    def test_diagonal_movement_cost(self):
        """Test that diagonal movements have correct cost."""
        start = Point(0, 0)
        goal = Point(1, 1)
        
        path = self.planner.find_path(start, goal)
        
        self.assertIsNotNone(path)
        # Should prefer diagonal movement for efficiency
        self.assertEqual(len(path), 2)
    
    def test_heuristic_function(self):
        """Test the heuristic function."""
        point1 = Point(0, 0)
        point2 = Point(3, 4)
        
        distance = self.planner.heuristic(point1, point2)
        expected = 5.0  # sqrt(3^2 + 4^2) = 5
        
        self.assertAlmostEqual(distance, expected, places=5)
    
    def test_neighbor_generation(self):
        """Test neighbor generation and cost calculation."""
        center = Point(5, 5)
        neighbors = self.planner.get_neighbors(center)
        
        # Should have 8 neighbors in empty workspace
        self.assertEqual(len(neighbors), 8)
        
        # Check that all neighbors are valid
        for neighbor, cost in neighbors:
            self.assertTrue(self.planner.is_valid_position(neighbor))
            self.assertGreater(cost, 0)
    
    def test_obstacle_blocks_neighbors(self):
        """Test that obstacles block neighbor generation."""
        center = Point(5, 5)
        
        # Add obstacles around the center
        self.planner.add_obstacle(4, 5)  # Left
        self.planner.add_obstacle(6, 5)  # Right
        self.planner.add_obstacle(5, 4)  # Up
        self.planner.add_obstacle(5, 6)  # Down
        
        neighbors = self.planner.get_neighbors(center)
        
        # Should have fewer neighbors due to obstacles
        self.assertLess(len(neighbors), 8)
    
    def test_path_optimality(self):
        """Test that the path is optimal (shortest possible)."""
        # Create a scenario where we know the optimal path
        start = Point(0, 0)
        goal = Point(2, 2)
        
        path = self.planner.find_path(start, goal)
        
        self.assertIsNotNone(path)
        # Optimal path should be diagonal: (0,0) -> (1,1) -> (2,2)
        # or (0,0) -> (2,2) if diagonal movement is allowed
        self.assertLessEqual(len(path), 3)
    
    def test_large_workspace_performance(self):
        """Test performance with larger workspace."""
        large_planner = RobotTrajectoryPlanner(50, 50)
        
        # Add some obstacles
        for i in range(0, 50, 5):
            large_planner.add_obstacle(i, 25)
        
        start = Point(0, 0)
        goal = Point(49, 49)
        
        start_time = time.time()
        path = large_planner.find_path(start, goal)
        end_time = time.time()
        
        self.assertIsNotNone(path)
        self.assertLess(end_time - start_time, 1.0)  # Should complete within 1 second
    
    def test_complex_obstacle_pattern(self):
        """Test with complex obstacle pattern."""
        # Create a maze-like pattern
        obstacles = [
            (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
            (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
            (6, 2), (6, 3), (6, 4), (6, 5), (6, 6),
            (8, 2), (8, 3), (8, 4), (8, 5), (8, 6)
        ]
        
        for x, y in obstacles:
            self.planner.add_obstacle(x, y)
        
        start = Point(0, 0)
        goal = Point(9, 9)
        
        path = self.planner.find_path(start, goal)
        
        self.assertIsNotNone(path)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
        
        # Verify path doesn't go through obstacles
        for point in path:
            self.assertNotIn(point, self.planner.obstacles)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_robot_simulation_scenario(self):
        """Test a realistic robot simulation scenario."""
        # Simulate a kitchen workspace
        planner = RobotTrajectoryPlanner(20, 15)
        
        # Add kitchen obstacles (counters, appliances)
        # Counter along the bottom
        for x in range(20):
            planner.add_obstacle(x, 0)
        
        # Island in the middle
        for x in range(8, 12):
            for y in range(6, 9):
                planner.add_obstacle(x, y)
        
        # Refrigerator on the right
        for x in range(18, 20):
            for y in range(1, 8):
                planner.add_obstacle(x, y)
        
        # Test multiple robot paths
        test_cases = [
            (Point(1, 14), Point(19, 14)),  # Top to top
            (Point(1, 14), Point(1, 1)),    # Top to bottom left
            (Point(19, 14), Point(15, 1)),  # Top right to bottom (avoiding refrigerator)
        ]
        
        for start, goal in test_cases:
            path = planner.find_path(start, goal)
            self.assertIsNotNone(path, f"Failed to find path from {start} to {goal}")
            self.assertEqual(path[0], start)
            self.assertEqual(path[-1], goal)


def run_performance_benchmark():
    """Run performance benchmarks."""
    print("Running performance benchmarks...")
    
    # Test different workspace sizes
    sizes = [(10, 10), (20, 20), (30, 30)]
    
    for width, height in sizes:
        planner = RobotTrajectoryPlanner(width, height)
        
        # Add random obstacles (10% of workspace)
        import random
        random.seed(42)  # For reproducible results
        
        num_obstacles = int(width * height * 0.1)
        for _ in range(num_obstacles):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            planner.add_obstacle(x, y)
        
        start = Point(0, 0)
        goal = Point(width - 1, height - 1)
        
        start_time = time.time()
        path = planner.find_path(start, goal)
        end_time = time.time()
        
        print(f"Workspace {width}x{height}: {end_time - start_time:.4f}s, "
              f"Path length: {len(path) if path else 'None'}")


if __name__ == "__main__":
    print("Running Robot Trajectory Planner Tests")
    print("=" * 50)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "=" * 50)
    print("Running Performance Benchmarks")
    print("=" * 50)
    
    # Run performance benchmarks
    run_performance_benchmark()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
