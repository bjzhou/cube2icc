import subprocess
import shutil
from pathlib import Path
import sys

def main():
    # Define directories
    base_dir = Path(".")
    cube_dir = base_dir / "cube"
    icc_dir = base_dir / "icc"
    
    # Check if cube directory exists
    if not cube_dir.exists():
        print(f"Error: Directory '{cube_dir}' does not exist.")
        sys.exit(1)
        
    # Create icc directory if it doesn't exist
    icc_dir.mkdir(exist_ok=True)
    
    # Find all .cube files in the cube directory
    cube_files = list(cube_dir.glob("*.cube"))
    
    if not cube_files:
        print(f"No .cube files found in {cube_dir}")
        return

    print(f"Found {len(cube_files)} cube files to process.")

    # Process each cube file
    for cube_file in cube_files:
        print(f"Processing: {cube_file.name}...")
        
        try:
            # Construct the command
            # Using uv run python as requested
            cmd = ["uv", "run", "python", "main.py", str(cube_file), "--gamma", "1"]
            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"  Successfully processed {cube_file.name}")
                # Print stdout if there's any relevant info (optional)
                # print(result.stdout)
            else:
                print(f"  Failed to process {cube_file.name}")
                print(f"  Error output:\n{result.stderr}")
                
        except Exception as e:
            print(f"  An error occurred while running command: {e}")

    # Move generated .icc files from cube/ to icc/
    # main.py generates .icc files in the same directory as the input file
    generated_icc_files = list(cube_dir.glob("*.icc"))
    
    if generated_icc_files:
        print(f"\nMoving {len(generated_icc_files)} generated ICC files to {icc_dir}...")
        for icc_file in generated_icc_files:
            target_path = icc_dir / icc_file.name
            try:
                shutil.move(str(icc_file), str(target_path))
                print(f"  Moved: {icc_file.name}")
            except Exception as e:
                print(f"  Failed to move {icc_file.name}: {e}")
    else:
        print("\nNo ICC files found to move.")

if __name__ == "__main__":
    main()
