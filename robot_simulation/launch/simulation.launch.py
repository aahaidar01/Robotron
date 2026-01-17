import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'robot_simulation'
    
    # 1. Path to our custom world and urdf
    world_file = os.path.join(
        get_package_share_directory(pkg_name), 'worlds', 'my_world.world')
    urdf_file = os.path.join(
        get_package_share_directory(pkg_name), 'urdf', 'my_robot.urdf')

    # 2. Launch Gazebo with our world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_file}.items(),
    )

    # 3. Spawn the Robot using our urdf
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'my_turtlebot', '-file', urdf_file, '-x', '0', '-y', '0', '-z', '0.1'],
        output='screen'
    )
    
    # 4. Robot State Publisher (Required for TF)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': True, 'robot_description': open(urdf_file).read()}]
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
    ])