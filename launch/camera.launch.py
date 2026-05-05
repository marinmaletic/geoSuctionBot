from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py'
            ])
        ]),
        launch_arguments={
            'enable_rgbd':                'true',
            'enable_sync':                'true',
            'align_depth.enable':         'true',
            'enable_color':               'true',
            'enable_depth':               'true',
            'pointcloud.enable':          'true',
            'spatial_filter.enable':      'true',
            'temporal_filter.enable':     'true',
            'hole_filling_filter.enable': 'false',
            'depth_module.depth_profile': '848x480x30',
            'rgb_camera.color_profile':   '848x480x30',
        }.items()
    )

    camera_node = Node(
        package='ur_suctionbot',
        executable='camera_node.py',
        name='camera_node',
        output='screen',
        parameters=[{
            'save_dir': '/data/captures'
        }]
    )

    return LaunchDescription([
        realsense_launch,
        camera_node
    ])
