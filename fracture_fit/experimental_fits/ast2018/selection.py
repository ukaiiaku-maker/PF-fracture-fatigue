"""Choose the retained material vector while preserving each fit stage."""
import json
from pathlib import Path


def retained_vector_path(run_dir:Path,scenario:str)->Path:
    stem=scenario.lower()
    candidates=[]
    if scenario=="FIGURE5_ALTERNATIVE":
        candidates.append(run_dir/f"best_1d_refined_{stem}_profile_seeded.json")
    candidates.extend((run_dir/f"best_1d_refined_{stem}.json",
                       run_dir/f"best_{stem}.json"))
    return next((path for path in candidates if path.exists()),candidates[-1])


def archive_superseded_final(run_dir:Path,scenario:str,new_x:list[float])->None:
    """Keep all old final-vector evidence before recomputing it at ``new_x``."""
    stem=scenario.lower()
    archive=run_dir/f"superseded_{stem}_before_profile_seeded"
    manifest=archive/"manifest.json"
    final_vector=run_dir/f"confirmation_{stem}/vector.json"
    if not manifest.exists():
        if not final_vector.exists():
            return
        old_x=json.loads(final_vector.read_text())["x"]
        if old_x==new_x:
            return
        archive.mkdir(parents=True,exist_ok=True)
        manifest.write_text(json.dumps({"scenario":scenario,"old_x":old_x,
                                        "new_x":new_x},indent=2)+"\n")
    elif json.loads(manifest.read_text())["new_x"]!=new_x:
        raise RuntimeError("superseded final archive targets a different vector")
    names=(f"confirmation_{stem}",f"confirmation_{stem}_endpoint_validation",
           f"ensemble_{stem}_reduced",f"ensemble_{stem}_spatial",
           f"barriers_{stem}",f"figures_{stem}",
           f"curve_support_{stem}.json",f"no_cleavage_outcomes_{stem}.json")
    for name in names:
        source=run_dir/name
        destination=archive/name
        if destination.exists():
            continue
        if source.exists():
            source.rename(destination)
