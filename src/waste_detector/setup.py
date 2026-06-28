from setuptools import find_packages, setup

package_name = 'waste_detector'

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
    maintainer='lauraluna',
    maintainer_email='lls6@cin.ufpe.br',
    description='Nó ROS2 para detecção e localização 3D de resíduos sólidos',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'waste_detector_node = waste_detector.waste_detector_node:main',
            'waste_navigator_node = waste_detector.waste_navigator_node:main',
        ],
    },
)
