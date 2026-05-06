from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node

import os


def generate_launch_description():

    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='false',
        description='Launch RViz2')

    gui_arg = DeclareLaunchArgument(
        'gui', default_value='false',
        description='Launch tkinter GUI')

    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ur_suctionbot'), 'launch', 'camera.launch.py'
            ])
        ])
    )

    suction_node = TimerAction(
        period=3.0,
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
            ),
            Node(
                package='ur_suctionbot',
                executable='gui_node.py',
                name='gui_node',
                output='screen',
                condition=IfCondition(LaunchConfiguration('gui')),
            ),
            Node(
                package='ur_suctionbot',
                executable='segmentation_node.py',
                name='segmentation_node',
                parameters=[{
                    'sam2_checkpoint': os.path.expanduser('~/sam2/checkpoints/sam2.1_hiera_tiny.pt'),
                    'gemini_model':    'gemini-robotics-er-1.6-preview',
                    'depth_scale':     0.001,
                }],
                output='screen',
            ),
        ]
    )

    return LaunchDescription([
        rviz_arg,
        gui_arg,
        camera_launch,
        suction_node,
    ])