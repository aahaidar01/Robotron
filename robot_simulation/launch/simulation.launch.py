import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction, RegisterEventHandler, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'robot_simulation'
    robot_pose_maze = ['-0.5', '-2.1', '0.1', '1.57'] #['-0.7', '-2.2', '0.1', '1.57']  # x, y, z, yaw
    
    # 1. Path to our custom world and urdf
    world_file = os.path.join(
        get_package_share_directory(pkg_name), 'worlds', 'maze_zig_zag.world')
    urdf_file = os.path.join(
        get_package_share_directory(pkg_name), 'urdf', 'dagu.urdf')

    # 2. Cleanup any zombie Gazebo processes first
    cleanup = ExecuteProcess(
        cmd=['bash', '-c', 
             'killall -9 gzserver gzclient 2>/dev/null; '
             'rm -rf /tmp/gazebo* ~/.gazebo/server-* 2>/dev/null; '
             'sleep 2; exit 0'],
        output='screen',
        name='gazebo_cleanup'
    )
    
    pkg_share_path = get_package_share_directory(pkg_name)
    gazebo_model_path = os.path.join(pkg_share_path, '..')
    
    set_gazebo_model_path = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=[os.environ.get('GAZEBO_MODEL_PATH', ''), ':', gazebo_model_path]
    )
    
    # 3. Launch Gazebo with our world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world_file,
            'gui': 'true',
        }.items(),
    )

    # 4. Spawn the Robot using our urdf
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'my_turtlebot', '-file', urdf_file, '-x', robot_pose_maze[0], '-y', robot_pose_maze[1], '-z', robot_pose_maze[2], '-Y', robot_pose_maze[3] ,'-timeout', '60'],
        output='screen'
    )
    
    # 5. Robot State Publisher (Required for TF)
    with open(urdf_file, 'r') as f:
        robot_description = f.read()
    
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': True, 'robot_description': robot_description}]
    )

    # 6. Use TimerAction to delay Gazebo launch until cleanup finishes
    delayed_gazebo = TimerAction(
        period=3.0,
        actions=[gazebo]
    )
    
    delayed_spawn = TimerAction(
        period=15.0,  # Give Gazebo time to fully start
        actions=[spawn_entity]
    )

    return LaunchDescription([
        set_gazebo_model_path,
        cleanup,
        robot_state_publisher,
        delayed_gazebo,
        delayed_spawn,
    ])