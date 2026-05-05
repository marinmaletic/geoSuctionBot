from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    # Include camera launch
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ur_suctionbot'), 'launch', 'camera.launch.py'
            ])
        ])
    )

    # Suction node — delayed to wait for camera
    suction_node = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='ur_suctionbot',
                executable='suction_node.py',
                name='suction_node',
                parameters=[{
                    'method':          'knn',
                    'threshold':        0.5,
                    'knn_k':            30,
                    'std_win':          0,
                    'cup_diameter_mm':  30.0,
                    'ransac_iters':     50,
                    'ransac_tol_mm':    3.0,
                    'depth_scale':      0.001,
                }],
                output='screen',
            )
        ]
    )

    return LaunchDescription([
        camera_launch,
        suction_node,
    ])