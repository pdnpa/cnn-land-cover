'''
Use by:

import loadpaths
user_paths_dict = loadpath.loadpaths()
'''

import os
import json
import getpass
from pathlib import Path

def loadpaths(username=None):
    '''Function that loads data paths from json file based on computer username'''

    ## Get json:
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    json_path = os.path.join(Path(__location__).parent, 'content/data_paths.json')
    json_path = str(json_path)
    path_repo = Path(__location__).parent
    path_repo = str(path_repo)
    
    if username is None:
        username = getpass.getuser()  # get username of PC account

    ## Load paths corresponding to username:
    with open(json_path, 'r') as config_file:
        config_info = json.load(config_file)
        if username in config_info.keys():
            user_paths_dict = config_info[username]['paths']  # extract paths from current user
            # Expand tildes in the json paths
            user_paths_dict = {k: str(v) for k, v in user_paths_dict.items()}
        else:
            user_paths_dict = config_info['new-username']['paths'] 
            print('WARNING: Username not found in data_paths.json. Using default paths for new-username.')

    dict_add_relative = {
      "pd_outline": "content/National_Park/National_Park.shp",
      "landscape_character_grid_path": "content/landscape_character_grid/Landscape_Character_Grid.shp",
      "evaluation_50tiles": "content/evaluation_sample_50tiles/evaluation_sample_50tiles.shp",
      "evaluation_50tiles_polygons": "content/evaluation_polygons/landscape_character_2022_FGH-override/landscape_character_2022_FGH-override.shp",
      "evaluation_50tiles": "content/evaluation_sample_50tiles/evaluation_sample_50tiles.shp",
      "evaluation_50tiles_polygons": "content/evaluation_polygons/landscape_character_2022_FGH-override/landscape_character_2022_FGH-override.shp"
    }

    for k, v in dict_add_relative.items():
        if k not in user_paths_dict.keys():     
            user_paths_dict[k] = os.path.join(path_repo, v)

    user_paths_dict['repo'] = path_repo

    return {k: os.path.expanduser(v) for k, v in user_paths_dict.items()}
