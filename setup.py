from setuptools import find_packages, setup

package_name = 'tb3_state_estimation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yigitf05',
    maintainer_email='yigitf05@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
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
