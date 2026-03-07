import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/andria/aml_project_ws/src/Robotron/install/robot_simulation'
