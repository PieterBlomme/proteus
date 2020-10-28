def read_class_names(class_file_name):
    """loads class name from a file"""
    names = {}
    with open(class_file_name, "r") as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip("\n")
    return names
