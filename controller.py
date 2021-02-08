from TFL_mng.tfl_mng import *
import xml.etree.ElementTree as ET
import os

import pickle
import matplotlib._png as png
import numpy as np

tree = ET.parse('play_list.xml')
root = tree.getroot()
# -------------------
tfl_points = [[np.array([117, 633]), np.array([366, 818]), np.array([114, 189]), np.array([178, 933]), np.array([199, 635]), np.array([ 335, 1118]), np.array([342, 541]), np.array([ 399, 1136]), np.array([412, 870]),
                np.array([417, 387])], [np.array([114, 632]), np.array([365, 819]), np.array([172, 934]), np.array([  74, 1014]), np.array([111, 187]), np.array([129,  77]), np.array([ 333, 1120]), np.array([340, 535]),
                np.array([ 398, 1138]), np.array([411, 870]), np.array([416, 387])], [np.array([365, 819]), np.array([126,  75]), np.array([169, 934]), np.array([198, 631]), np.array([337, 531]), np.array([ 332, 1123]),
                np.array([ 398, 1140]), np.array([416, 386]), np.array([450, 140])], [np.array([217, 631]), np.array([166, 933]), np.array([233, 850]), np.array([271, 744]), np.array([ 331, 1125]), np.array([336, 529]),
                np.array([365, 818]), np.array([396, 689]), np.array([ 398, 1142]), np.array([411, 872]), np.array([416, 385]), np.array([451, 124])], [np.array([194, 593]), np.array([163, 931]), np.array([127,  71]), 
                np.array([ 330, 1127]), np.array([337, 517]), np.array([366, 819]), np.array([ 399, 1143]), np.array([413, 872]), np.array([417, 383]), np.array([453, 107])], [np.array([194, 589]), np.array([160, 932]),
                np.array([111, 255]), np.array([114, 317]), np.array([127,  69]), np.array([ 330, 1129]), np.array([337, 513]), np.array([366, 820]), np.array([ 399, 1146]), np.array([414, 873]), np.array([419, 383]),
                np.array([455,  91]), np.array([ 611, 1699])]]
tfl_points = [[np.array([point[1], point[0]]) for point in arr] for arr in tfl_points]
# -------------------



for clip in root.iter('clip'):
    path = clip.find('path').text
    pkl = clip.find('pkl').text
    images_tree = clip.find('images')
    start_name = images_tree.attrib['start_name']
    end_name = images_tree.attrib['end_name']

    with open(os.path.join(path, pkl), 'rb') as pklfile:
        data = pickle.load(pklfile, encoding='latin1')
    focal = data['flx']
    pp = data['principle_point']
    tfl_mng = TflMng(pp, focal)


    for image in images_tree:
        seq = image.attrib['seq']
        image_name = start_name + seq + end_name
        EM = data['egomotion_' + str(int(seq)-1) + '-' + seq] if image is not images_tree[0] else None
        tfl_mng.init_frame(os.path.join(path, image_name), EM)
        tfl_mng.run_frame()
        

        



        
