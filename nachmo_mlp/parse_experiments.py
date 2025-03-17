#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 11:30:53 2025

@author: andrey
"""

import yaml

def load_experiments(file_path):
    """Reads a YAML file and returns the data as a dictionary."""
    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return None


def parse_experiments(file_path):
    data = load_experiments(file_path)
    """Stores each level of the YAML data in separate lists."""
    if not data or "mechanisms" not in data:
        print("Invalid or missing data.")
        return [], [], []

    mechanisms_list = list(data["mechanisms"].keys())
    settings_list = []
    options_dict = {}

    for mech, details in data["mechanisms"].items():
        settings_list.append({
            "mechanism": mech,
            "current_epoch": details["current_epoch"],
            "n_steps": details["n_steps"],
            "slices": details["slices"],
            "path_to_data": details["path_to_data"],
            "rollout_length": details["rollout_length"],
            "tries": details["tries"],
            "random_starts": details["random_starts"]
        })
        options_dict[mech] = details["options"]

    return mechanisms_list, settings_list, options_dict
