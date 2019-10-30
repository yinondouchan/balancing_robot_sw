import bluetooth
import struct

ROBOT_MAC_ADDRESS = '00:18:E4:36:1D:89'

tunable_parameters = {
    'ANGLE_PID_P': {'tag': 0, 'length': 4, 'type': 'float32'},                # inner PID loop angle proportional component
    'ANGLE_PID_ANG_VEL_P': {'tag': 1, 'length': 4, 'type': 'float32'},        # inner PID loop angular velocity proportional component
    'VEL_PID_P': {'tag': 2, 'length': 4, 'type': 'float32'},                  # outer PID loop velocity proportional
    'VEL_PID_I': {'tag': 3, 'length': 4, 'type': 'float32'},                  # outer PID loop velocity integral
    'VEL_PID_D': {'tag': 4, 'length': 4, 'type': 'float32'},                  # outer PID loop velocity derivative
    'VEL_LPF_TC': {'tag': 5, 'length': 4, 'type': 'float32'},                 # velocity low-pass filter time constant [microseconds]
    'BALANCE_VEL_LIMIT': {'tag': 6, 'length': 4, 'type': 'int32'},          # balancing algorithm velocity limit [full-steps per second]
    'VEL_PID_D2': {'tag': 7, 'length': 4, 'type': 'float32'},                 # additional derivative factor
    'CURRENT_LIMIT': {'tag': 8, 'length': 4, 'type': 'float32'},
}


def send_parameters(bluetooth_connection, parameter_names, parameter_values):
    """
    send parameters to robot through bluetooth
    :param bluetooth_connection:
    :param parameter_names: list of parameter names according to the tunable_parameters field
    :param parameter_values: list of parameter values corresponding to the parameter names
    :return: nothing. YOU GET NOTHING!
    """
    tags = [tunable_parameters[parameter_name]['tag'] for parameter_name in parameter_names]
    lengths = [tunable_parameters[parameter_name]['length'] for parameter_name in parameter_names]
    types = [tunable_parameters[parameter_name]['type'] for parameter_name in parameter_names]

    for tag, length, value, type_ in zip(tags, lengths, parameter_values, types):
        # write to serial
        # param command starts with P and then is followed by a tag byte, a length byte and value bytes
        param_byte_array = b'P'
        param_byte_array += struct.pack("b", length)


        if type == 'float32' or type == 'int32':
            # store 4 bytes of value byte by byte
            param_byte_array += struct.pack("f", value)

        # send parameter command
        bluetooth_connection.send(param_byte_array)


port = 1
connection = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
connection.connect((ROBOT_MAC_ADDRESS, port))

send_parameters(connection, ['ANGLE_PID_P'], [200.0])

