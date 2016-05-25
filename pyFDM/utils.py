import ConfigParser


def get_strings_section(config, section):
    options = config.options(section)
    section_dict = {}    
    for option in options:
        section_dict[option] = config.get(section, option)
    return section_dict


def get_numbers_section(config, section):
    options = config.options(section)
    section_dict = {}    
    for option in options:
        if option == "nx" or option == "tc":
            section_dict[option] = config.getint(section, option)
        else:
            section_dict[option] = config.getfloat(section, option)
    return section_dict
    

def read_config(fname):
    """
    Reads config.cfg file.
        
    Reads configuration file and sets up parameters for the simulation.
    
    :param fname: Filename of the configuration file.
    """
    config = ConfigParser.SafeConfigParser()
    config.optionxform = str 
    config.read(fname)
    # Files
    files = get_strings_section(config, 'Files')
    # Geoemtry 
    # get number of section containing "Geometry"
    arteries = get_numbers_section(config, 'Arteries')
    # Simulation 
    sim = get_numbers_section(config, 'Simulation')
    return files, arteries, sim


def read_csv(fname):
    f = open(fname, 'r')
    lines = f.readlines()
    f.close()
    u = []
    t = []    
    for l in lines:
        data = l.split(',')
        t.append(float(data[0]))
        u.append(float(data[1]))
    return u, t
    
    
def periodic(t, T):
    while t/T > 1.0:
        t = t - T
    return t
    
    
def extrapolate(x0, x, y):
    return y[0] + (y[1]-y[0]) * (x0 - x[0])/(x[1] - x[0])