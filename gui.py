#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import sys
from pathlib import Path
import os

# Ensure we can import from main.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from main import CUBEReader, ICCParser, ICCWriter, PRESETS
except ImportError as e:
    print(f"Error importing modules from main.py: {e}")
    sys.exit(1)

class Cube2IccApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CUBE to ICC Converter")
        self.root.geometry("600x500")


        # Variables
        self.cube_path = tk.StringVar()
        self.selected_preset = tk.StringVar()
        self.gamma_val = tk.DoubleVar(value=1.0)
        
        # Profile Discovery
        self.profile_dir, self.profile_files = self.find_profiles_dir()
        self.base_icc_path = tk.StringVar()
        
        # Set default value if found
        if self.profile_files:
            # Try to find a sensible default or just use the first one
            default_profile = next((f for f in self.profile_files if "Sony" in f), self.profile_files[0])
            self.base_icc_path.set(default_profile)
        else:
            # Fallback default
            self.base_icc_path.set('/Applications/Capture One.app/Contents/Frameworks/ImageProcessing.framework/Versions/A/Resources/Profiles/Input/SonyA7CM2-ProStandard.icm')

        self.setup_ui()

    def find_profiles_dir(self):
        """
        Attempt to find the Capture One Input Profiles directory.
        Returns: (directory_path_obj, [list_of_filenames]) or (None, [])
        """
        found_dir = None
        
        if sys.platform == 'darwin': # macOS
            mac_path = Path('/Applications/Capture One.app/Contents/Frameworks/ImageProcessing.framework/Versions/A/Resources/Profiles/Input/')
            if mac_path.exists():
                found_dir = mac_path
                
        elif sys.platform == 'win32': # Windows
            # Search C:/Program Files/ for Capture One / Phase One
            pf = Path(os.environ.get('PROGRAMFILES', 'C:/Program Files'))
            
            # Simple heuristic search
            possible_roots = list(pf.glob('Capture One*')) + list(pf.glob('Phase One*'))
            # Sort by name (likely version) reversed to get latest
            possible_roots.sort(key=lambda p: p.name, reverse=True)
            
            for root in possible_roots:
                # Look for Color Profiles/DSLR
                target = root / 'Color Profiles' / 'DSLR' # Logic from user request: [Capture One]/Color Profiles/DSLR
                # Note: User request: [Capture One or Capture One {version}]/Color Profiles/DSLR
                # But sometimes it might be deeper? Let's check the requested path structure carefully.
                # "C:/Program Files/[CaptureOne或Phase One]/[Capture One或Capture One {version}]/Color Profiles/DSLR"
                # This suggests a nested structure: Program Files -> Phase One -> Capture One 23 -> Color Profiles -> DSLR
                # Or direct: Program Files -> Capture One 23 -> Color Profiles -> DSLR
                
                # Let's try to walk a bit if needed, or check common paths.
                # Basic check 1: direct under root
                check1 = root / 'Color Profiles' / 'DSLR'
                if check1.exists():
                    found_dir = check1
                    break
                    
                # Basic check 2: nested one level (e.g. Program Files/Phase One/Capture One 20/...)
                for child in root.iterdir():
                    if child.is_dir() and 'Capture One' in child.name:
                        check2 = child / 'Color Profiles' / 'DSLR'
                        if check2.exists():
                            found_dir = check2
                            break
                if found_dir: break

        files = []
        if found_dir and found_dir.exists():
            # List .icm / .icc files
            files = [f.name for f in found_dir.glob('*.icm')] + [f.name for f in found_dir.glob('*.icc')]
            files.sort()
            
        return found_dir, files

    def setup_ui(self):
        # Main Frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. Base ICC Selection
        ttk.Label(main_frame, text="Base ICC Profile:").grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        icc_frame = ttk.Frame(main_frame)
        icc_frame.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        
        if self.profile_files:
            # Use Combobox if profiles found
            self.icc_cb = ttk.Combobox(icc_frame, textvariable=self.base_icc_path, values=self.profile_files)
            self.icc_cb.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
            # Keep browse button to allow override
            ttk.Button(icc_frame, text="Browse...", command=self.browse_icc).pack(side=tk.RIGHT)
        else:
            # Standard Entry
            ttk.Entry(icc_frame, textvariable=self.base_icc_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
            ttk.Button(icc_frame, text="Browse...", command=self.browse_icc).pack(side=tk.RIGHT)

        # 2. Cube LUT Selection
        ttk.Label(main_frame, text="Input LUT (.cube):").grid(row=2, column=0, sticky="w", pady=(0, 5))
        
        cube_frame = ttk.Frame(main_frame)
        cube_frame.grid(row=3, column=0, sticky="ew", pady=(0, 15))
        
        ttk.Entry(cube_frame, textvariable=self.cube_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(cube_frame, text="Browse...", command=self.browse_cube).pack(side=tk.RIGHT)

        # 3. Settings (Preset & Gamma)
        settings_frame = ttk.Labelframe(main_frame, text="Settings", padding="10")
        settings_frame.grid(row=4, column=0, sticky="ew", pady=(0, 15))

        # Preset
        ttk.Label(settings_frame, text="Target Preset:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        preset_cb = ttk.Combobox(settings_frame, textvariable=self.selected_preset, state="readonly")
        preset_cb['values'] = list(PRESETS.keys())
        if PRESETS:
            preset_cb.current(0)
        preset_cb.grid(row=0, column=1, sticky="ew", padx=(0, 20))

        # Gamma
        ttk.Label(settings_frame, text="Gamma:").grid(row=0, column=2, sticky="w", padx=(0, 10))
        ttk.Entry(settings_frame, textvariable=self.gamma_val, width=10).grid(row=0, column=3, sticky="w")
        
        # 4. Generate Button
        self.generate_btn = ttk.Button(main_frame, text="Generate ICC Profile", command=self.start_generation)
        self.generate_btn.grid(row=5, column=0, pady=(0, 15), ipady=5, sticky="ew")

        # 5. Log Area
        ttk.Label(main_frame, text="Status Log:").grid(row=6, column=0, sticky="w", pady=(0, 5))
        self.log_area = scrolledtext.ScrolledText(main_frame, height=10, state='disabled', font=("Menlo", 11))
        self.log_area.grid(row=7, column=0, sticky="nsew")

        # Grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(7, weight=1)

    def browse_icc(self):
        initial_dir = self.profile_dir if self.profile_dir else "/"
        filename = filedialog.askopenfilename(title="Select Base ICC Profile", initialdir=initial_dir, filetypes=[("ICC Profiles", "*.icm *.icc")])
        if filename:
            self.base_icc_path.set(filename)

    def browse_cube(self):
        filename = filedialog.askopenfilename(title="Select CUBE LUT", filetypes=[("Cube LUTs", "*.cube")])
        if filename:
            self.cube_path.set(filename)

    def log(self, message):
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')

    def start_generation(self):
        thread = threading.Thread(target=self.run_process)
        thread.start()

    def run_process(self):
        self.generate_btn.config(state='disabled')
        try:
            self._process()
        except Exception as e:
            self.log(f"❌ Critical Error: {e}")
        finally:
            self.generate_btn.config(state='normal')

    def _process(self):
        # Resolve Base ICC Path
        raw_icc_input = self.base_icc_path.get()
        base_icc = raw_icc_input
        
        # If input is just a filename and we have a directory discovered, join them
        if self.profile_dir and (raw_icc_input in self.profile_files):
            base_icc = self.profile_dir / raw_icc_input
        
        # If raw input is an absolute path that exists, use it
        elif Path(raw_icc_input).exists():
            base_icc = Path(raw_icc_input)
            
        # Check validity
        if not base_icc or not Path(base_icc).exists():
             # One last try: if user typed a filename manually but it's in the dir
             if self.profile_dir and (self.profile_dir / raw_icc_input).exists():
                 base_icc = self.profile_dir / raw_icc_input
             else:
                self.log(f"❌ Error: Invalid Base ICC path: {raw_icc_input}")
                return

        cube_file = self.cube_path.get()
        preset_name = self.selected_preset.get()
        gamma = self.gamma_val.get()
        
        if not cube_file or not Path(cube_file).exists():
            self.log("❌ Error: Invalid CUBE file path")
            return

        self.log("-" * 40)
        self.log(f"Starting conversion...")
        self.log(f"Base ICC: {Path(base_icc).name}")
        self.log(f"LUT: {Path(cube_file).name}")
        
        # Get Preset Details
        preset = PRESETS.get(preset_name)
        if not preset:
            self.log(f"❌ Error: Unknown preset {preset_name}")
            return
            
        target_gamut = preset['gamut']
        target_curve = preset['curve']
        self.log(f"Preset '{preset_name}': {target_gamut} / {target_curve}")

        # 1. Parse ICC
        parser = ICCParser(base_icc)
        base_profile_data = None
        if parser.parse():
            base_profile_data = (parser.header, parser.tags)
        else:
            self.log("❌ Failed to parse Base ICC")
            return

        # 2. Read Cube
        reader = CUBEReader(cube_file)
        if not reader.read():
            self.log("❌ Failed to read CUBE file")
            return

        if reader.size != 33:
            self.log(f"Resampling LUT from {reader.size} to 33...")
            reader.data = reader.resample(33)

        # 3. Determine Output Path
        base_stem = Path(base_icc).stem
        if '-' in base_stem:
            camera_model = base_stem.rpartition('-')[0]
        else:
            camera_model = base_stem
            
        lut_name = Path(cube_file).stem
        out_filename = f"{camera_model}-{lut_name}.icc"
        out_path = Path(cube_file).parent / out_filename

        self.log(f"Output: {out_filename}")

        # 4. Generate
        # We assume defaults for lut_output_gamut/curve as 'sRGB' just like main.py defaults
        writer = ICCWriter(
            reader.data, 
            reader.title, 
            base_icc, 
            base_profile_data,
            target_gamut=target_gamut,
            target_curve=target_curve,
            lut_output_gamut='sRGB',
            lut_output_curve='sRGB',
            gamma=gamma
        )

        # Redirect stdout to log (hacky but effective for wrapping main.py logic)
        # Or just trust our own logging. main.py prints a lot.
        # We can capture stdout if we really want, but for now let's just run it.
        
        success = writer.create_profile(str(out_path))
        
        if success:
            self.log(f"✅ Success! Saved to {out_path}")
            
            # Auto-install if profile directory is known
            if self.profile_dir and self.profile_dir.exists():
                try:
                    import shutil
                    dest_path = self.profile_dir / out_filename
                    shutil.copy2(out_path, dest_path)
                    self.log(f"📂 Installed to: {dest_path}")
                    self.log("💡 Please restart Capture One to see the new profile.")
                except Exception as e:
                     self.log(f"⚠️ Failed to install to Capture One folder: {e}")
        else:
            self.log("❌ Generation failed during processing.")

if __name__ == "__main__":
    root = tk.Tk()
    # Attempt to improve high-DPI awareness on macOS
    try:
        from ctypes import cdll
        # This might not work on all macOS setups or might be ignored, but worth a try or just rely on tk defaults
        pass
    except:
        pass
        
    app = Cube2IccApp(root)
    root.mainloop()
