import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'tb3_state_estimation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yiğit Fanoşçu',
    maintainer_email='yigitf05@hotmail.com',
    description='EKF fusing wheel odometry and gyro for a TurtleBot3 in Gazebo, '
                'with a wheel-slip noise relay and ground-truth evaluation tools.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ekf_node = tb3_state_estimation.ekf_node:main',
            'odom_noise_relay = tb3_state_estimation.odom_noise_relay:main',
        ],
    },
)
