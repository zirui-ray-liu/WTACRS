import json
import sys

_true_set = {'yes', 'true', 't', 'y', '1'}
_false_set = {'no', 'false', 'f', 'n', '0'}

def str2bool(value):
    if isinstance(value, str):
        value = value.lower()
        if value in _true_set:
            return True
        if value in _false_set:
            return False

    return None

json_file = sys.argv[1]
update_json_file = sys.argv[-1]

with open(json_file, "r") as f:
    d = json.load(f)

attribute = sys.argv[2]
if attribute != "":

    attribute_type = sys.argv[3]
    attribute_value = sys.argv[4]
    print(attribute_type, attribute, attribute_value)

    if attribute_type == "bool":
        attribute_value = str2bool(attribute_value)
    elif attribute_type == "list":
        list_of_str = attribute_value.split(",")
        attribute_value = [int(x) for x in list_of_str]
    else:
        attribute_value = eval(attribute_type)(attribute_value)

    d[attribute] = attribute_value


if attribute == "output_dir":
    print(d)

# print(attribute, attribute_value)

# print(update_json_file)
with open(update_json_file, "w") as f:
    json.dump(d, f, indent=0)