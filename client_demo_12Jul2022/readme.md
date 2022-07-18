Directory: 2022-Power-Grids/client_demo_12Jul2022

This directory is designated for the scripts, data files and learning models for the client demo on 12 Jul 2022.

How to use the scripts:
- "input_handling_tool_demo.py"
    - This is the tool for handling the input events, converting the signals to PSR images, 
    and using the CNN model
    to generate predictions.
    - Usage: python3 input_handling_tool_demo.py [OPTIONS] FILEPATH
    - Help: python3 input_handling_tool_demo.py --help

- "dashboard_demo.py"
    - This is the script for running the web dashboard.
    - Usage: panel serve dashboard_demo.py --show

- This demo relies on the 500 data samples in CSV format in the "event_data" directory.
