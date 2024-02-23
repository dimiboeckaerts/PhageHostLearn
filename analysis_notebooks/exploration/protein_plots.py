"""
PROTEIN PLOTS

Created on Wed 16/06/21

@author: dimiboeckaerts
"""

# 0 - LIBRARIES
# --------------------------------------------------
# %%
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from numpy.linalg import norm


# 1 -  FUNCTIONS
# --------------------------------------------------
#  %%
def protein_architecture_plot(proteins, domains, locations, label_dict=[], count_threshold=0, save_fig=False):
    """
    Plots the different architectures (combinations of modules) for a given
    set of proteins, domains and their locations.

    Input:
    - proteins: list of protein sequences to plot
    - domains: list of lists with the domain names for each protein
    - locations: list of lists of tuples with the location of each corresponding domain
    - label_dict: optional dict with categories for labels {domain1: labelx, domain2: labely, ...}
    - count_threshold: threshold under which not to plot the domains, based on the number of occurrences
    - save_fig: option to save the figure
    """
    # initiations
    y_place = 0
    protein_lengths = [len(x) for x in proteins]

    #if plot_all:
    #    # loop over the proteins
    #    y_count = max(5, int(len(proteins)/4))
    #    y_box = int(y_count*0.6)
    #    fig, ax = plt.subplots(figsize=(8, y_count))
    #    x_legend = min(400, max(protein_lengths))
    #
    #    for i, current_protein in enumerate(proteins):
    #        # get current items
    #        y_place += y_count
    #        current_domains = domains[i]
    #        current_locations = locations[i]
    #        backbone_length = len(current_protein)
    #        protein_lengths.append(backbone_length)
    #
    #        # plot backbone
    #        backbone = plt.Rectangle((1, y_place), backbone_length, y_count*0.1, fc='grey')
    #        ax.add_patch(backbone)
    #
    #        # loop over domains
    #        for j, dom in enumerate(current_domains):
    #            # plot each domain at correct location
    #            loc = current_locations[j]
    #            patch = mpatch.FancyBboxPatch((loc[0], y_place-(y_box/2)), loc[1]-loc[0], y_box, 
    #                boxstyle='Round, pad=0.2, rounding_size=0.8', fc=cdict[dom], label=dom)
    #            ax.add_patch(patch)
    #    ax.set_xlim(0, max(protein_lengths)+x_legend)

    #else:
    unique_combos = [list(x) for x in set(tuple(x) for x in domains)] # get unique combos
    domain_counts = [domains.count(x) for x in unique_combos] # count unique combos
    sorted_unique_combos = [(x,y) for y, x in sorted(zip(domain_counts, unique_combos))] # sort
    sorted_unique_combos = [combo for combo in sorted_unique_combos if combo[1] > count_threshold] # delete under thres

    # give all unique domains or labels a separate color
    merged_domains = [dom for current_domains in sorted_unique_combos for dom in current_domains[0]]
    unique_domains = list(set(merged_domains))
    if len(label_dict) > 0:
        label_dict = dict([(dom, label_dict[dom]) for dom in label_dict.keys() if dom in unique_domains])
        unique_labels = list(set(label_dict.values()))
        cmap = plt.cm.turbo(np.linspace(0.0, 1.0, len(unique_labels)))
        cdict = dict([(label, cmap[i]) for i, label in enumerate(unique_labels)])
    else:
        cmap = plt.cm.turbo(np.linspace(0.0, 1.0, len(unique_domains)))
        cdict = dict([(dom, cmap[i]) for i, dom in enumerate(unique_domains)])

    # set up plot and params
    y_count = max(5, int(len(sorted_unique_combos)/4))
    y_box = int(y_count*0.6)
    x_count = min(140, max(protein_lengths)/3)
    x_legend = min(800, max(protein_lengths))
    fig, ax = plt.subplots(figsize=(8,y_count))

    # loop over unique combos and plot
    for i, current in enumerate(sorted_unique_combos):
        current_domains = current[0]
        current_count = current[1]
        y_place += y_count
        index = domains.index(current_domains)
        current_protein = proteins[index]
        current_locations = locations[index]
        backbone_length = len(current_protein)
        protein_lengths.append(backbone_length)

        # plot backbone
        backbone = plt.Rectangle((x_count, y_place), backbone_length, y_count*0.1, fc='grey')
        ax.add_patch(backbone)
        ax.annotate(str(current_count), xy=(1, y_place-(y_box/2)))

        # loop over domains
        for j, dom in enumerate(current_domains):
            # plot each domain at correct location
            loc = current_locations[j]

            if len(label_dict) > 0:
                current_label = label_dict[dom]
                current_color = cdict[current_label]
            else:
                current_label = dom
                current_color = cdict[dom]
            patch = mpatch.FancyBboxPatch((x_count+loc[0], y_place-(y_box/2)), loc[1]-loc[0], y_box, 
                boxstyle='Round, pad=0.2, rounding_size=0.8', fc=current_color, label=current_label)
            ax.add_patch(patch)
    ax.set_xlim(0, x_count+max(protein_lengths)+x_legend)

    ax.set_ylim(0, y_place+max(10,y_count))
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.axis('off')
    ax.set_title('Protein domain architectures', size=14)

    if save_fig:
        fig.savefig('protein_architecture_plot.png', dpi=400)

    fig.show()

    return


def protein_dimensions_ratio(name, mmcif_file, output='ratio'):
    """
    This function computes/estimates the length and width of a given 
    protein structure.

    Input:
    - name: string of the name of the structure
    - mmcif_file: mmCIF file of the structure
    - output: type of output to return ('ratio', 'dims' or 'points')
    """
    # initiate
    parser = MMCIFParser()
    structure = parser.get_structure(name, mmcif_file)
    #model = structure[0]
    coordinates = []

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    x,y,z = atom.get_coord()
                    coordinates.append((x,y,z))
    coordinates =  np.array(coordinates)

    # compute the points that are on a convex hull
    hull = ConvexHull(coordinates)
    hullpoints = coordinates[hull.vertices,:]

    # compute the maximal distance between points on the hull
    hdist = cdist(hullpoints, hullpoints, metric='seuclidean')
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
    P1, P2 = hullpoints[bestpair[0]], hullpoints[bestpair[1]]
    struct_length = norm(P2-P1)

    # compute the distances perpendicular to the direction of the maximal distance
    struct_width = 0
    P3 = 0
    for point in coordinates:
        new_width = 2*norm(np.cross(P2-P1, P1-point))/norm(P2-P1)
        if new_width > struct_width:
            struct_width = new_width
            P3 = point

    if output == 'ratio':
        return struct_length/struct_width
    elif output == 'dims':
        return struct_length, struct_width
    elif output == 'points':
        return coordinates, hullpoints, P1, P2, P3


def protein_dimensions_scatter(name_list, file_list, save_fig=False):
    """
    This function plots the dimensions of a list of protein structures

    Input:
    - name_list: list of names or labels to plot
    - file_list: list of mmCIF files of protein structures
    - save_fig: option to save the figure (default: False)
    """
    # initiations
    fig, ax = plt.subplots(figsize=(10,8))
    #unique_names = list(set(name_list))
    #cmap = plt.cm.rainbow(np.linspace(0.0, 1.0, len(unique_names)))

    for i, name in enumerate(name_list):
        file = file_list[i]
        struct_length, struct_width = protein_dimensions_ratio(name, file, output='dims')
        ax.scatter(struct_length, struct_width, label=name)

    ax.set_xlabel('length')
    ax.set_ylabel('width')
    ax.legend()
    plt.show()

    if save_fig:
        fig.save_fig('structure_dimensions_scatter.png', dpi=400)

    return

def points_3D_plot(points, P1, P2, P3):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for hull in points:
        ax.scatter(hull[0], hull[1], hull[2], c='black')
    ax.scatter(P1[0], P1[1], P1[2], c='red', s=50)
    ax.scatter(P2[0], P2[1], P2[2], c='red', s=50)
    ax.scatter(P3[0], P3[1], P3[2], c='green', s=50)
    plt.show()
    return


# 2 -  EXAMPLE
# --------------------------------------------------
# %%
domains = [['BppU', 'C24', 'C77'], ['fiber', 'klm', 'c15'], ['BppU', 'C24', 'C77'], ['pop', 'lol', 'n98']]
proteins = [''.join(random.choices(['5', '6', '7', '9'], k=30))]*4
locations = [[(4, 7), (9, 11), (20, 25)], [(1, 8), (10, 15), (20, 27)], [(2, 8), (11, 17), (25, 28)], [(4, 7), (9, 11), (20, 25)]]
unique = [list(x) for x in set(tuple(x) for x in domains)]
protein_architecture_plot(proteins, domains, locations)

# %%
RBPs = pd.read_csv('/Users/dimi/GoogleDrive/PhD/4_PHAGEHOST_LEARNING/42_DATA/RBPbase_250621.csv', 
        converters={'N_blocks': eval, 'C_blocks': eval, 'N_ranges': eval, 'C_ranges': eval})
RBPsmini = RBPs.iloc[:100,:]

# %%
label_dict = {'Phage_T7_tail': 'Phage_T7_tail', 'Tail_spike_N': 'Tail_spike_N', 'Prophage_tail': 'Prophage_tail', 
                'BppU_N': 'BppU_N', 'Mtd_N': 'Mtd_N', 'Head_binding': 'Head_binding', 
                'DUF3751': 'DUF3751', 'End_N_terminal': 'End_N_terminal', 'phage_tail_N': 'phage_tail_N', 
                'Prophage_tailD1': 'Prophage_tailD1', 'DUF2163': 'DUF2163', 'Phage_fiber_2': 'Phage_fiber_2',
                'Lipase_GDSL_2': 'Lipase_GDSL_2', 'Pectate_lyase_3': 'Pectate_lyase_3', 'gp37_C': 'gp37_C', 
                'Beta_helix': 'Beta_helix', 'Gp58': 'Gp58', 'End_beta_propel': 'End_beta_propel', 
                'End_tail_spike': 'End_tail_spike', 'End_beta_barrel': 'End_beta_barrel', 'PhageP22-tail': 'PhageP22-tail', 
                'Phage_spike_2': 'Phage_spike_2', 'gp12-short_mid': 'gp12-short_mid', 'Collar': 'Collar', 
                'Peptidase_S74': 'Peptidase_S74', 'Phage_fiber_C': 'Phage_fiber_C', 'S_tail_recep_bd': 'S_tail_recep_bd', 
                'CBM_4_9': 'CBM_4_9', 'DUF1983': 'DUF1983', 'DUF3672': 'DUF3672', 
                'phage_RBP_N1': 'N-terminal anchor (constructed)', 'phage_RBP_N4': 'N-terminal anchor (constructed)', 'phage_RBP_N26': 'N-terminal anchor (constructed)', 
                'phage_RBP_N28': 'N-terminal anchor (constructed)', 'phage_RBP_N34': 'N-terminal anchor (constructed)', 
                'phage_RBP_N45': 'N-terminal anchor (constructed)', 
                'phage_RBP_C2': 'C-terminal domain (constructed)', 'phage_RBP_C10': 'C-terminal domain (constructed)', 'phage_RBP_C24': 'C-terminal domain (constructed)', 
                'phage_RBP_C43': 'C-terminal domain (constructed)', 'phage_RBP_C59': 'C-terminal domain (constructed)', 'phage_RBP_C60': 'C-terminal domain (constructed)', 
                'phage_RBP_C62': 'C-terminal domain (constructed)', 'phage_RBP_C67': 'C-terminal domain (constructed)', 'phage_RBP_C79': 'C-terminal domain (constructed)', 
                'phage_RBP_C97': 'C-terminal domain (constructed)', 'phage_RBP_C111': 'C-terminal domain (constructed)', 'phage_RBP_C115': 'C-terminal domain (constructed)', 
                'phage_RBP_C120': 'C-terminal domain (constructed)', 'phage_RBP_C126': 'C-terminal domain (constructed)', 'phage_RBP_C138': 'C-terminal domain (constructed)', 
                'phage_RBP_C143': 'C-terminal domain (constructed)', 'phage_RBP_C157': 'C-terminal domain (constructed)', 'phage_RBP_C164': 'C-terminal domain (constructed)',
                'phage_RBP_C175': 'C-terminal domain (constructed)', 
                'phage_RBP_C180': 'C-terminal domain (constructed)', 'phage_RBP_C205': 'C-terminal domain (constructed)', 'phage_RBP_C217': 'C-terminal domain (constructed)', 'phage_RBP_C220': 'C-terminal domain (constructed)', 'phage_RBP_C221': 'C-terminal domain (constructed)',
                'phage_RBP_C223': 'C-terminal domain (constructed)', 'phage_RBP_C234': 'C-terminal domain (constructed)', 'phage_RBP_C235': 'C-terminal domain (constructed)', 'phage_RBP_C237': 'C-terminal domain (constructed)', 'phage_RBP_C249': 'C-terminal domain (constructed)', 
                'phage_RBP_C259': 'C-terminal domain (constructed)', 'phage_RBP_C267': 'C-terminal domain (constructed)', 'phage_RBP_C271': 'C-terminal domain (constructed)', 'phage_RBP_C277': 'C-terminal domain (constructed)', 'phage_RBP_C281': 'C-terminal domain (constructed)', 
                'phage_RBP_C292': 'C-terminal domain (constructed)', 'phage_RBP_C293': 'C-terminal domain (constructed)', 'phage_RBP_C296': 'C-terminal domain (constructed)', 'phage_RBP_C300': 'C-terminal domain (constructed)', 'phage_RBP_C301': 'C-terminal domain (constructed)', 
                'phage_RBP_C319': 'C-terminal domain (constructed)', 'phage_RBP_C320': 'C-terminal domain (constructed)', 'phage_RBP_C321': 'C-terminal domain (constructed)', 'phage_RBP_C326': 'C-terminal domain (constructed)', 'phage_RBP_C331': 'C-terminal domain (constructed)', 
                'phage_RBP_C337': 'C-terminal domain (constructed)', 'phage_RBP_C338': 'C-terminal domain (constructed)', 'phage_RBP_C340': 'C-terminal domain (constructed)', 
                'other': 'Other', '': 'Other'}

proteins = RBPsmini.protein_seq
domains = [RBPsmini.N_blocks[i]+RBPsmini.C_blocks[i] for i in range(RBPsmini.shape[0])]
locations = [RBPsmini.N_ranges[i]+RBPsmini.C_ranges[i] for i in range(RBPsmini.shape[0])]
protein_architecture_plot(proteins, domains, locations, label_dict=label_dict, count_threshold=5)

# %%
# FINAL PLOT
proteins = RBPs.protein_seq
domains = [RBPs.N_blocks[i]+RBPs.C_blocks[i] for i in range(RBPs.shape[0])]
locations = [RBPs.N_ranges[i]+RBPs.C_ranges[i] for i in range(RBPs.shape[0])]
protein_architecture_plot(proteins, domains, locations, label_dict=label_dict, count_threshold=5)

# %%
# PROTEIN STRUCTURES
file = '/Users/Dimi/Downloads/2xgf.cif'
name = "2xGF"

# %%
# PROTEIN STRUCTURES: 3D PLOTS / dimensions plots

RBPnames = ['2xgf', '1ocy', '2c3f', '4a0t', '4qnl', '4uxg', '5yvq', '6f45', '6iab']
nonRBPnames = ['5eut', '5jbl', '5v7e', '6a9b', '6xc1', '7d7d']
RBPfiles = ['/Users/Dimi/Downloads/structures/'+name+'.cif' for name in RBPnames]
nonRBPfiles = ['/Users/Dimi/Downloads/structures/'+name+'.cif' for name in nonRBPnames]
files = RBPfiles+nonRBPfiles
names = RBPnames+nonRBPnames

#protein_dimensions_scatter(names, files, save_fig=False)
for i, name in enumerate(RBPnames[2]):
    file = RBPfiles[i]
    points, hp, P1, P2, P3 = protein_dimensions_ratio(name, file, output='points')
    points_3D_plot(hp, P1, P2, P3)


# %%
parser = MMCIFParser()
file = '/Users/Dimi/Downloads/structures/2c3f.cif'
name = "2c3f"
points, hp, P1, P2, P3 = protein_dimensions_ratio(name, file, output='points')
coordinates = []
for model in structure:
    for chain in model:
        for residue in chain:
            try:
                ca = residue['CA']
                x,y,z = ca.get_coord()
                coordinates.append((x,y,z))
            except KeyError:
                pass

            #for atom in residue:
            #    x,y,z = atom.get_coord()
            #    coordinates.append((x,y,z))
coordinates =  np.array(coordinates)
points_3D_plot(coordinates, P1, P2, P3)
# %%
