import os
from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue # <-- New import for Humble

def generate_launch_description():
    pkg_name = 'my_rover_description'
    
    xacro_file = PathJoinSubstitution([
        FindPackageShare(pkg_name),
        'urdf',
        'rover.urdf.xacro'
    ])

    # Evaluate the xacro command as a string for Humble
    robot_description_content = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': True 
        }]
    )

    spawn_rover = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'lunar_rover',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '1.0', 
        ],
        output='screen'
    )

    return LaunchDescription([
        robot_state_publisher,
        spawn_rover
    ])