import json
import os
from typing import Optional, Dict


def load_json(json_file_fullname: str) -> Dict:
    with open(json_file_fullname, 'r') as f:
        run_config = json.load(f)

    return run_config


def save_json(json_data: Dict, json_name: str, json_path: Optional[str] = '') -> None:
    json_name = json_name if json_name.endswith('.json') else "{}.json".format(json_name)
    json_fullname = os.path.join(json_path, json_name)

    with open(json_fullname, 'w') as f:
        json.dump(json_data, f)

    print("Saved JSON of len {} at: {}".format(len(json_data), json_fullname))