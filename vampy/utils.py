from configparser import ConfigParser
import matplotlib.pylab as plt
import numpy as np


def get_strings_section(config, section):
    """
    Get config file options from section containing strings.
    
    :param config: ConfigParser object.
    :param section: Name of the section to be read.
    """
    options = config.options(section)
    section_dict = {}    
    for option in options:
        section_dict[option] = config.get(section, option)
    return section_dict


def get_numbers_section(config, section):
    """
    Get config file options from section containing numbers.
    
    :param config: ConfigParser object.
    :param section: Name of the section to be read.
    """
    options = config.options(section)
    section_dict = {}    
    for option in options:
        if option in ["tc", "ntr", "depth"]:
            section_dict[option] = config.getint(section, option)
        else:
            try:
                section_dict[option] = config.getfloat(section, option)
            except ValueError:
                opt_list = config.get(section, option).split(',')
                section_dict[option] = np.array([
                                            float(opt) for opt in opt_list])
    return section_dict
    

def read_config(fname):
    """
    Reads config.cfg file.
        
    Reads configuration file and sets up parameters for the simulation.
    
    :param fname: Filename of the configuration file.
    """
    config = ConfigParser()
    config.optionxform = str 
    config.read(fname)
    # Files
    files = get_strings_section(config, 'Files')
    # Arteries
    arteries = get_numbers_section(config, 'Arteries')
    # Simulation 
    sim = get_numbers_section(config, 'Simulation')
    return files, arteries, sim


def periodic(t, T):
    """
    Returns equivalent time of the first period if more than one period is simulated.
    
    :param t: Time.
    :param T: Period length.
    """
    while t/T > 1.0:
        t = t - T
    return t
    
    
def extrapolate(x0, x, y):
    """
    Returns extrapolated data point given two adjacent data points.
    
    :param x0: Data point to be extrapolated to.
    :param x: x-coordinates of known data points.
    :param y: y-coordinates of known data points.
    """
    return y[0] + (y[1]-y[0]) * (x0 - x[0])/(x[1] - x[0])