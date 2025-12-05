import os
import crabnet.utils.utils as crab_utils

def patch_optimizer_nuclear():
    file_path = crab_utils.__file__
    print(f"Targeting library file: {file_path}")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    patches_made = 0
    
    for line in lines:
        # Target the line we fixed in v4
        target_v4 = "if float(weight_norm) == 0 or float(adam_norm) == 0:"
        # Also target the original line just in case v4 wasn't applied or was reverted
        target_original = "if weight_norm == 0 or adam_norm == 0:"
        
        if target_v4 in line or target_original in line:
            indent = line[:line.find("if")]
            
            # THE FIX: .cpu().item() forces the value to system RAM before Python checks it.
            # This bypasses the buggy Metal driver completely for this check.
            new_line = indent + "if weight_norm.cpu().item() == 0 or adam_norm.cpu().item() == 0:\n"
            
            new_lines.append(new_line)
            patches_made += 1
            print("Fixed: Optimizer Zero-Check (Nuclear CPU Move)")
            
        else:
            new_lines.append(line)

    if patches_made == 0:
        print("No match found. Please check if the file matches expected content.")
    else:
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
        print(f"SUCCESS! Applied {patches_made} fixes.")

if __name__ == "__main__":
    patch_optimizer_nuclear()