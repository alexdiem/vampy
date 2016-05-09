import ConfigParser


def get_strings_section(config, section):
    options = config.options(section)
    section_dict = {}    
    for option in options:
        param[option] = config.get(section, option)
    return section_dict


def get_numbers_section(config, section):
    options = config.options(section)
    section_dict = {}    
    for option in options:
        if option == "nx":
            param[option] = config.getint(section, option)
        else:
            param[option] = config.getfloat(section, option)
    return section_dict
    

def read_config(fname):
    """
    Reads config.cfg file.
        
    Reads configuration file and sets up parameters for the simulation.
    
    :param fname: Filename of the configuration file.
    """
    config = ConfigParser.RawConfigParser()
    config.read(fname)
    # Files
    files = get_strings_section(config, 'Files')
    # Geoemtry 
    # get number of section containing "Geometry"
    arteries = get_numbers_section(config, 'Arteries')
    # Simulation 
    sim = get_numbers_section(config, 'Simulation')
    return files, arteries, sim
