from setuptools import setup

package_name = 'lidar_scan_clustering'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='임정근',
    maintainer_email='jglim@inha.ac.kr',
    description='2D LiDAR (/scan) 기반 클러스터링 실습 패키지 (TurtleBot3 Simulation)',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # ros2 run lidar_scan_clustering scan_cluster
            'scan_cluster = lidar_scan_clustering.scan_cluster_node:main',
            'pf_cluster_avoid = lidar_scan_clustering.pf_cluster_avoid_node:main',
            'pf_click_goal_avoid = lidar_scan_clustering.pf_click_goal_avoid_node:main',
            'planner_pf_astar = lidar_scan_clustering.planner_pf_astar_node:main',
        ],
    },
)
