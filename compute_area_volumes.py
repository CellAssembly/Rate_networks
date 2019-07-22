#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:27:19 2019

@author: joshs
"""

labels = np.load('/mnt/md0/data/opt/annotation_volume_10um_by_index.npy')

structure_tree = pd.read_csv('/mnt/md0/data/opt/template_brain/ccf_structure_tree_2017.csv')

# %%

acronyms = ('VISp','VISl','VISal','VISrl','VISpm','VISam')

layers = ('1','2/3','4','5','6a','6b')

column_labels = ['structure','area']


structures = []
volumes = []

for acronym in acronyms:
    
    for layer in layers:
        
        full_acronym = acronym + layer
        
        print(full_acronym)
        
        index = structure_tree[structure_tree['acronym'] == full_acronym].index.values[0] + 1
        
        volume = np.sum(labels == index) / 1e6
        
        print(' ' + str(volume))
        
        structures.append(full_acronym)
        volumes.append(volume)
# %%        
acronyms = ('LP','LGd')

for acronym in acronyms:

    print(acronym)
    
    matches = structure_tree[structure_tree['acronym'].str.match(acronym)]
    
    for index, row in matches.iterrows():
        
        full_acronym = row['acronym']

        volume = np.sum(labels == index + 1) / 1e6
        
        print(' ' + str(volume))
        
        structures.append(full_acronym)
        volumes.append(volume)
# %%
        
     
        # %%
df = pd.DataFrame(data = {'structure': structures, 'volume': volumes})

df.to_csv('/home/joshs/Dropbox/AIBS/Figures/visual_area_volumes.csv')