"""
    Brief description
    
    Author: Edilson A. Correa Junior <edilsonacjr@gmail.com>
"""


import numpy as np
from io import BytesIO
from subprocess import Popen, PIPE
import sys

#@profile
def to_xnet(g, file_name, names=True):

    """
    Adapted from Filipi's code (https://github.com/filipinascimento)

    Convert igraph object to a .xnet format string. This string
    can then be written to a file or used in a pipe. The purpose
    of this function is to have a fast means to convert a graph
    without any attributes to the .xnet format.

    Parameters
    ----------
    g : igraph.Graph
        Input graph.

    Returns
    -------
    None
    """
    with open(file_name, 'w') as xnet_file:
        N = g.vcount()
        xnet_file.write('#vertices '+str(N)+' nonweighted\n')
        xnet_file.write('#v “Class” s\n')
        if names:
            for v in g.vs['name']:
                xnet_file.write('"' + str(v) + '"' + '\n')
        xnet_file.write('#edges weighted undirected\n')
        #edList = g.get_edgelist()
        #edList = [(f, t, w) for (f, t), w in zip(edList, g.es['weight'])]
        #ed_list = g.get_edgelist()
        #we_list = g.es['weight']
        #for i in range(len(ed_list)):
        #    xnet_file.write('%d %d %f' % (ed_list[i][0], ed_list[i][1], we_list[i]))
        for e, w in zip(g.es, g.es['weight']):
            xnet_file.write('%d %d %f\n' % (e.source, e.target, w))
