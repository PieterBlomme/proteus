import os
from pathlib import Path
template_path = Path(__file__).parent / 'templates'

def main():
    print (template_path)
    os.system(f'cookiecutter {template_path}')