"""
This script converts .npy q_table file into .h (header) file 
to be imported as header file in main arduino script.
"""

import numpy as np
import os

def convert_npy_to_h(npy_filepath, h_filepath):
    """Converts a numpy Q-table into a C++ header file."""
    try:
        q_table = np.load(npy_filepath)
    except Exception as e:
        print(f"Error loading {npy_filepath}: {e}")
        return

    rows, cols = q_table.shape
    print(f"Loaded Q-table with shape: {rows}x{cols}")

    with open(h_filepath, 'w') as f:
        # Include guards to prevent multiple definition errors
        f.write("#ifndef Q_TABLE_H\n")
        f.write("#define Q_TABLE_H\n\n")
        
        # Array declaration
        f.write(f"const float Q_TABLE[{rows}][{cols}] = {{\n")
        
        # Write array data
        for i in range(rows):
            # Format to 4 decimal places to save memory/file size while keeping precision
            row_strs = [f"{val:.4f}" for val in q_table[i]]
            row_line = "    {" + ", ".join(row_strs) + "}"
            
            # Add a comma to all rows except the last one
            if i < rows - 1:
                row_line += ",\n"
            else:
                row_line += "\n"
            
            f.write(row_line)
            
        # Close array and include guards
        f.write("};\n\n")
        f.write("#endif // Q_TABLE_H\n")

    print(f"Successfully exported to {h_filepath}!")
    print("You can now move this file into your Arduino sketch folder.")

if __name__ == "__main__":
    # Point this to your actual Q-table file
    input_file = "q_table_latest.npy" 
    output_file = "q_table.h"
    
    if os.path.exists(input_file):
        convert_npy_to_h(input_file, output_file)
    else:
        print(f"File '{input_file}' not found. Check the path and try again.")