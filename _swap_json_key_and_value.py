import json

with open("filtered_96.json", "r") as read_file:
    data = json.load(read_file)

# Use a dictionary comprehension to swap keys and values
swapped_dict = {v: k for k, v in data.items()}

with open("filtered_96_swapped.json", "w") as write_file:
    json.dump(swapped_dict, write_file)
