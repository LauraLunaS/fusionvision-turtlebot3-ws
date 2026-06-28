from setuptools import setup
package_name = 'obstacle_avoider'
setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lauraluna',
    maintainer_email='lauraluna@todo.todo',
    description='Desvio de obstáculos com LiDAR',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'avoider_node = obstacle_avoider.avoider_node:main',
        ],
    },
)
