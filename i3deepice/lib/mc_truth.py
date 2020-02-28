from icecube import dataclasses, dataio, icetray, MuonGun
from icecube.icetray import I3Units
import icecube.MuonGun
import numpy as np

decode = {0: 'Skimming',
          1: 'Starting Cascade',
          2: 'Through-Going Track',
          3: 'Starting Track',
          4: 'Stopping Track',
          5: 'Double-Bang',
          6: 'Stopping Tau',
          7: 'Glashow Hadronic Cascade',
          8: 'Glashow Track',
          9: 'Glashow Tau',
          11: 'Passing Track',
          21: 'Several Muons',
          22: 'Through-Going Bundle',
          23: 'Stopping Bundle',
          100: 'Unclassified',
          101: 'Unclassified'}

nu_pdg = [12, 14, 16, -12, -14, -16]

def classify_wrapper(p_frame, surface=None, gcdfile=None):
    if is_data(p_frame):
        return True
    if gcdfile is None:
        if 'I3Geometry' in p_frame.keys():
            gcdfile = p_frame['I3Geometry']
        else:
            raise ValueError('Please add a Geometry File to the input or remove the add_truth argument')
    if surface is None:
        surface = icecube.MuonGun.ExtrudedPolygon.from_I3Geometry(gcdfile, padding=0)
    if 'I3MCWeightDict' in p_frame.keys():
        classify(p_frame, surface=surface, gcdfile=gcdfile)
    elif 'MuonWeight' in p_frame.keys():
        classify_muongun(p_frame, surface=surface, gcdfile=gcdfile)
    elif 'CorsikaWeightMap' in p_frame.keys():
        classify_corsika(p_frame, surface=surface, gcdfile=gcdfile)
    else:
        print('Not sure how to classify')
    return

def find_all_neutrinos(p_frame):
    if is_data(p_frame):
        return True
    I3Tree = p_frame['I3MCTree']
    # find first neutrino as seed for find_particle
    for p in I3Tree.get_primaries():
        if p.pdg_encoding in nu_pdg:
            break
    all_nu = [i for i in crawl_neutrinos(p, I3Tree, plist=[]) if len(i) > 0]
    return all_nu[-1][0]


def crawl_neutrinos(p, I3Tree, level=0, plist = []):
    if len(plist) < level+1:
        plist.append([])
    if (p.is_neutrino) & np.isfinite(p.length):
        plist[level].append(p)
    children = I3Tree.children(p)
    if len(children) < 10:
        for child in children:
            crawl_neutrinos(child, I3Tree, level=level+1, plist=plist)
    return plist


def is_data(frame):
    if ('I3MCWeightDict' in frame) or ('MuonWeight' in frame)  or ('CorsikaWeightMap' in frame) or\
            ('MCPrimary' in frame) or ('I3MCTree' in frame):
        return False
    else:
        return True

def has_signature(p, surface):
    intersections = surface.intersection(p.pos, p.dir)
    if p.is_neutrino:
        return -1
    if not np.isfinite(intersections.first):
        return -1
    if p.is_cascade:
        if intersections.first <= 0 and intersections.second > 0:
            return 0  # starting event
        else:
            return -1  # no hits
    elif p.is_track:
        if intersections.first <= 0 and intersections.second > 0:
            return 0  # starting event
        elif intersections.first > 0 and intersections.second > 0:
            if p.length <= intersections.first:
                return -1  # no hit
            elif p.length > intersections.second:
                return 1  # through-going event
            else:
                return 2  # stopping event
        else:
            return -1

# Generation of the Classification Label
def classify(p_frame, gcdfile=None, surface=None):
    pclass = 101 # only for security
    I3Tree = p_frame['I3MCTree']
    neutrino = find_all_neutrinos(p_frame)
    children = I3Tree.children(neutrino)
    p_types = [np.abs(child.pdg_encoding) for child in children]
    p_strings = [child.type_string for child in children]
    p_frame.Put("visible_nu", neutrino)
    IC_hit = np.any([((has_signature(tp, surface) != -1) & np.isfinite(tp.length)) for tp in children])
    if p_frame['I3MCWeightDict']['InteractionType'] == 3 and (len(p_types) == 1 and p_strings[0] == 'Hadrons'):
        pclass = 7  # Glashow Cascade
    else:
        if (11 in p_types) or (p_frame['I3MCWeightDict']['InteractionType'] == 2):
            if IC_hit:
                pclass = 1  # Cascade
            else:
                pclass = 0 # Uncontainced Cascade
        elif (13 in p_types):
            mu_ind = p_types.index(13)
            p_frame.Put("visible_track", children[mu_ind])
            if not IC_hit:
                pclass = 11 # Passing Track
            elif p_frame['I3MCWeightDict']['InteractionType'] == 3:
                if has_signature(children[mu_ind], surface) == 0:
                    pclass = 8  # Glashow Track
            elif has_signature(children[mu_ind], surface) == 0:
                pclass = 3  # Starting Track
            elif has_signature(children[mu_ind], surface) == 1:
                pclass = 2  # Through Going Track
            elif has_signature(children[mu_ind], surface) == 2:
                pclass = 4  # Stopping Track
        elif (15 in p_types):
            tau_ind = p_types.index(15)
            p_frame.Put("visible_track", children[tau_ind])
            if not IC_hit:
                pclass = 12 # uncontained tau something...
            else:
                # consider to use the interactiontype here...
                if p_frame['I3MCWeightDict']['InteractionType'] == 3:
                    pclass =  9  # Glashow Tau
                else:
                    had_ind = p_strings.index('Hadrons')
                    try:
                        tau_child = I3Tree.children(children[tau_ind])[-1]
                    except:
                        tau_child = None
                    if tau_child:
                        if np.abs(tau_child.pdg_encoding) == 13:
                            if has_signature(tau_child, surface) == 0:
                                pclass = 3  # Starting Track
                            if has_signature(tau_child, surface) == 1:
                                pclass = 2  # Through Going Track
                            if has_signature(tau_child, surface) == 2:
                                pclass = 4  # Stopping Track
                        else:
                            if has_signature(children[tau_ind], surface) == 0 and has_signature(tau_child, surface) == 0:
                                pclass = 5  # Double Bang
                            if has_signature(children[tau_ind], surface) == 0 and has_signature(tau_child, surface) == -1:
                                pclass = 3  # Starting Track
                            if has_signature(children[tau_ind], surface) == 2 and has_signature(tau_child, surface) == 0:
                                pclass = 6  # Stopping Tau
                            if has_signature(children[tau_ind], surface) == 1:
                                pclass = 2  # Through Going Track
                    else: # Tau Decay Length to large, so no childs are simulated
                        if has_signature(children[tau_ind], surface) == 0:
                            pclass = 3 # Starting Track
                        if has_signature(children[tau_ind], surface) == 1:
                            pclass = 2  # Through Going Track
                        if has_signature(children[tau_ind], surface) == 2:
                            pclass = 4  # Stopping Track
        else:
            pclass = 100 # unclassified
    #print('Classification: {}'.format(pclass))
    p_frame.Put("classification_truth_id", icetray.I3Int(pclass))
    p_frame.Put("classification_truth", dataclasses.I3String(decode[pclass]))
    return


def classify_muongun(p_frame, gcdfile=None, surface=None, primary_key='MCPrimary'):
    p = p_frame[primary_key]
    if has_signature(p, surface) == 1:
        pclass = 2  # Through Going Track
    elif has_signature(p, surface) == 2:
        pclass = 4  # Stopping Track
    else:
        pclass = 0
    p_frame.Put("classification_truth_id", icetray.I3Int(pclass))
    p_frame.Put("classification_truth", dataclasses.I3String(decode[pclass]))
    p_frame.Put("visible_track", p)
    return


def classify_corsika(p_frame, gcdfile=None, surface=None):
    mu_list = []
    I3Tree = p_frame['I3MCTree']
    primaries = I3Tree.get_primaries()
    for p in primaries:
        tlist = []
        find_muons(p, I3Tree, surface, plist=tlist)
        mu_list.append(tlist[-1])

    if len(np.concatenate(mu_list)) == 0:
        pclass = 11 # Passing Track
    elif len(mu_list)>1:
        pclass = 21 # several events
        clist = np.concatenate(mu_list)
        energies = np.array([calc_depositedE_single_p(p, I3Tree, surface) for p in clist])
        inds = np.argsort(energies)
        p_frame.Put("visible_track", clist[inds[-1]])

    else:
        if len(mu_list[0]) >1:
            energies = np.array([calc_depositedE_single_p(p, I3Tree, surface) for p in mu_list[0]])
            mu_signatures = np.array([has_signature(p, surface) for p in mu_list[0]])
            if np.any(mu_signatures==1):
                pclass = 22 # Through Going Bundle
            else:
                pclass = 23 # Stopping Muon Bundle
            inds = np.argsort(energies)
            p_frame.Put("visible_track", mu_list[0][inds[-1]])
        else:
            if has_signature(mu_list[0][0], surface) == 2:
                pclass = 4 # Stopping Track
            else:
                pclass = 2 # Through Going Track
            p_frame.Put("visible_track", mu_list[0][0])
    p_frame.Put("classification_truth_id", icetray.I3Int(pclass))
    p_frame.Put("classification_truth", dataclasses.I3String(decode[pclass]))
    return
