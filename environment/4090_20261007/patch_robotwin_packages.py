from pathlib import Path
import re
import sysconfig

site = Path(sysconfig.get_paths()["purelib"])

loader = site / "sapien/wrapper/urdf_loader.py"
text = loader.read_text()
updated = re.sub(r'open\(([^,\n]+), "r"\)', r'open(\1, "r", encoding="utf-8")', text)
updated = updated.replace('urdf_file[:-4] + "srdf"', 'urdf_file[:-4] + ".srdf"')
if updated == text and 'encoding="utf-8"' not in text:
    raise RuntimeError(f"Expected SAPIEN file-read pattern missing: {loader}")
loader.write_text(updated)

planner = site / "mplib/planner.py"
text = planner.read_text()
old = "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:"
new = "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:"
if old not in text and new not in text:
    raise RuntimeError(f"Expected mplib condition missing: {planner}")
planner.write_text(text.replace(old, new))

print("SAPIEN and mplib patches verified")
