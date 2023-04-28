import os

def camel_to_snake(file_name):
    """
    Converts CamelCase strings to snake_case
    """
    snake_name = ''
    for i, c in enumerate(file_name):
        if i > 0 and c.isupper() and file_name[i-1].islower():
            snake_name += '_'
        snake_name += c.lower()
    return snake_name

# Path to the directory containing the files to rename
dir_path = './tables/'

for file_name in os.listdir(dir_path):
    # Skip any directories in the directory
    if os.path.isdir(os.path.join(dir_path, file_name)):
        continue
    # Convert the file name from CamelCase to snake_case
    new_name = camel_to_snake(file_name)
    # Rename the file with the new name
    os.rename(os.path.join(dir_path, file_name), os.path.join(dir_path, new_name))

