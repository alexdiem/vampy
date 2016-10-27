# -*- coding: utf-8 -*-

from vampy.utils import *


eps = 1e-5


def equal(a, b):
    return True if abs(a-b) < eps else False


def setup_config():
    config = ConfigParser.SafeConfigParser()
    config.optionxform = str 
    config.read("test_param.cfg")
    return config


def test_get_strings_section():
    config = setup_config()
    files = get_strings_section(config, 'Files')
    assert files['inlet'] == 'inlet.csv'
    assert len(files) == 1
    
    
def test_get_numbers_section():
    config = setup_config()
    arteries = get_numbers_section(config, 'Arteries')
    simulation = get_numbers_section(config, 'Simulation')    
    assert arteries['R'] == 0.0037
    assert arteries['a'] == 0.91
    assert arteries['b'] == 0.58
    assert len(arteries) == 3
    assert simulation['nx'] == 40
    assert simulation['T'] == 0.917
    assert len(simulation) == 2
    
    
def test_read_config():
    files, arteries, simulation = read_config("test_param.cfg")
    assert files['inlet'] == 'inlet.csv'
    assert len(files) == 1
    assert arteries['R'] == 0.0037
    assert arteries['a'] == 0.91
    assert arteries['b'] == 0.58
    assert len(arteries) == 3
    assert simulation['nx'] == 40
    assert simulation['T'] == 0.917
    assert len(simulation) == 2
    
    
def test_periodic():
    T = 3.0
    assert equal(periodic(5.0, T), 2.0)
    assert equal(periodic(3.0, T), 3.0)
    assert equal(periodic(3.5, T), 0.5)
    assert equal(periodic(7.2, T), 1.2)
    assert equal(periodic(2.5, T), 2.5)
    
    
def test_extrapolate():
    assert extrapolate(2, [0,1], [0,1]) == 2.0
    assert extrapolate(2, [0,1], [0,4]) == 8.0
    assert extrapolate(-2, [0,-1], [0,4]) == 8.0