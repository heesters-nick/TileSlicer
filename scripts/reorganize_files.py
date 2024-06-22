from pathlib import Path

# Set the directory containing the files
base_directory = Path(
    '/home/heesters/projects/def-sfabbro/heesters/data/unions/catalogs/GAaP/UNIONS2000'
)

# Iterate over each file in the directory
for file in base_directory.iterdir():
    if file.is_file() and file.name.endswith('_ugriz_photoz_ext.cat'):
        # Extract the directory name from the filename
        # Assuming the directory name is the part before the first underscore '_'
        directory_name = file.name.split('_')[0]
        # Create a Path object for the directory
        directory_path = base_directory / directory_name

        # Create the directory if it does not exist
        directory_path.mkdir(exist_ok=True)

        # Move the file to the new directory
        file.rename(directory_path / file.name)

print('Files have been organized into subdirectories.')
