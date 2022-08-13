from setuptools import setup

setup(
    name='input_handling_tool',
    version='0.1.0',
    py_modules=['input_handling_tool'],
    install_requires=[
        'Click',
    ],
    entry_points={
        'console_scripts': [
            'handleinput = input_handling_tool:main',
        ],
    },
)