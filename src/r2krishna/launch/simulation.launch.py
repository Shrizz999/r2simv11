import os
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch import LaunchService

def generate_launch_description():
    # Attempt to get the share directory, with a fallback for local development
    try:
        pkg_r2krishna = get_package_share_directory('r2krishna')
        pkg_arena_viz = get_package_share_directory('arena_viz')
        pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    except PackageNotFoundError:
        # Fallback: if not installed, look relative to this launch file's location
        # This assumes your structure is src/r2krishna/launch/simulation.launch.py
        pkg_r2krishna = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # You may need to adjust these paths if arena_viz is in a different folder
        pkg_arena_viz = os.path.abspath(os.path.join(pkg_r2krishna, '..', 'arena_viz'))
        pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim') # Usually globally installed

    # Resource paths
    install_dir = os.path.dirname(pkg_r2krishna)
    gz_resource_path = f"{install_dir}:{pkg_arena_viz}/models"
    
    # Updated path logic to handle both source and install structures
    urdf_file = os.path.join(pkg_r2krishna, 'urdf', 'auto_3.urdf')
    world_file = os.path.join(pkg_arena_viz, 'worlds', 'arena.world')
    rviz_config = os.path.join(pkg_arena_viz, 'rviz', 'arena.rviz')

    if not os.path.exists(urdf_file):
        raise FileNotFoundError(f"URDF not found at: {urdf_file}")

    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    lidar_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='lidar_static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'lidar_link', 'auto_3/base_footprint/lidar_3d'],
        output='screen'
    )

    return LaunchDescription([
        SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=gz_resource_path),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
            launch_arguments={'gz_args': f'-r {world_file}'}.items(),
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc, 'use_sim_time': True}]
        ),

        Node(
            package='ros_gz_sim',
            executable='create',
            arguments=['-topic', 'robot_description', '-name', 'auto_3', '-x', '-2.0', '-y', '0.5', '-z', '1.0', '-Y', '0'],
            output='screen'
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            parameters=[{'qos_overrides./tf_static.publisher.durability': 'transient_local'}],
            arguments=[
                '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
                '/cmd_vel_front@geometry_msgs/msg/Twist@gz.msgs.Twist', 
                '/scan_3d/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked', 
                '/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image',
                '/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
                '/belly_camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image', 
                '/imu@sensor_msgs/msg/Imu@gz.msgs.IMU',
                '/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V',
                '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
                '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model' 
            ],
            output='screen'
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': True}]
        ),
        
        lidar_tf_node
    ])

if __name__ == '__main__':
    ls = LaunchService()
    ls.include_launch_description(generate_launch_description())
    ls.run()
