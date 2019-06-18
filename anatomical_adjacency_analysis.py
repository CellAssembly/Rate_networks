import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import pylab


params = {'legend.fontsize': 22,
          'axes.labelsize': 22,
          'axes.titlesize': 22,
          'xtick.labelsize': 22,
          'ytick.labelsize': 22}
pylab.rcParams.update(params)



###### For data from Sam & Corrbett  #######
AdjMat = np.load('Anatomical_adj_mat/ccMat_rowNorm.npy')
areas = np.array(['POR', 'LI', 'LM', 'V1', 'AL', 'RL', 'AM', 'PM', 'ACA'])

# plt.matshow(AdjMat)
out_degrees = np.nansum(AdjMat, axis = 1)
in_degrees  = np.nansum(AdjMat, axis = 0)
ratio = in_degrees/out_degrees

inds = np.array([3, 2, 4, 5, 7, 6])
print inds
print areas[inds]
print ratio[inds]
color_list = ['#E3CF57', 'green', 'purple', 'blue', 'brown', 'red']

fig = plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.imshow(AdjMat[2:8][:,2:8], cmap = 'magma')
plt.xticks(np.arange(len(inds)), areas[2:8])
plt.yticks(np.arange(len(inds)), areas[2:8])
plt.xlabel('Target')
plt.ylabel('Source')
plt.subplot(1,2,2)
plt.scatter(np.arange(len(inds)), ratio[inds], color = color_list, lw = 10)
plt.xticks(np.arange(len(inds)), areas[inds])
plt.ylabel('in-degree/out-degree')
plt.tight_layout()
# plt.savefig("adjacency_matrix/anatomical_degree_quantification.pdf", format = 'pdf', dpi = 1000)


##################################################
##### For whole brain data from Jennifer
##################################################
AdjMat = pd.DataFrame.from_csv('Anatomical_adj_mat/normalized_connection_density.csv')
V1_L1  = np.where(AdjMat.index == 'VISp1')[0][0]
V1_L23 = np.where(AdjMat.index == 'VISp2/3')[0][0]
V1_L4  = np.where(AdjMat.index == 'VISp4')[0][0]
V1_L5  = np.where(AdjMat.index == 'VISp5')[0][0]
V1_L6A = np.where(AdjMat.index == 'VISp6a')[0][0]
V1_L6B = np.where(AdjMat.index == 'VISp6b')[0][0]

AL1  = np.where(AdjMat.index == 'VISal1')[0][0]
AL23 = np.where(AdjMat.index == 'VISal2/3')[0][0]
AL4  = np.where(AdjMat.index == 'VISal4')[0][0]
AL5  = np.where(AdjMat.index == 'VISal5')[0][0]
AL6A = np.where(AdjMat.index == 'VISal6a')[0][0]
AL6B = np.where(AdjMat.index == 'VISal6b')[0][0]

AM1  = np.where(AdjMat.index == 'VISam1')[0][0]
AM23 = np.where(AdjMat.index == 'VISam2/3')[0][0]
AM4  = np.where(AdjMat.index == 'VISam4')[0][0]
AM5  = np.where(AdjMat.index == 'VISam5')[0][0]
AM6A = np.where(AdjMat.index == 'VISam6a')[0][0]
AM6B = np.where(AdjMat.index == 'VISam6b')[0][0]

PM1  = np.where(AdjMat.index == 'VISpm1')[0][0]
PM23 = np.where(AdjMat.index == 'VISpm2/3')[0][0]
PM4  = np.where(AdjMat.index == 'VISpm4')[0][0]
PM5  = np.where(AdjMat.index == 'VISpm5')[0][0]
PM6A = np.where(AdjMat.index == 'VISpm6a')[0][0]
PM6B = np.where(AdjMat.index == 'VISpm6b')[0][0]

LM1  = np.where(AdjMat.index == 'VISl1')[0][0]
LM23 = np.where(AdjMat.index == 'VISl2/3')[0][0]
LM4  = np.where(AdjMat.index == 'VISl4')[0][0]
LM5  = np.where(AdjMat.index == 'VISl5')[0][0]
LM6A = np.where(AdjMat.index == 'VISl6a')[0][0]
LM6B = np.where(AdjMat.index == 'VISl6b')[0][0]

RL1  = np.where(AdjMat.index == 'VISrl1')[0][0]
RL23 = np.where(AdjMat.index == 'VISrl2/3')[0][0]
RL4  = np.where(AdjMat.index == 'VISrl4')[0][0]
RL5  = np.where(AdjMat.index == 'VISrl5')[0][0]
RL6A = np.where(AdjMat.index == 'VISrl6a')[0][0]
RL6B = np.where(AdjMat.index == 'VISrl6b')[0][0]
#AdjMat.iloc[145][144]

Vis_AdjMat = np.zeros((12,12))
Area_order = ['V1s', 'V1d', 'LMs', 'LMd', 'ALs', 'ALd', 'RLs', 'RLd', 'PMs', 'PMd','AMs', 'AMd']
V1_s_inds = np.arange(V1_L1, V1_L4 + 1)
V1_d_inds = np.arange(V1_L5, V1_L6B + 1)

LM_s_inds = np.arange(LM1, LM4 + 1)
LM_d_inds = np.arange(LM5, LM6B + 1)

AL_s_inds = np.arange(AL1, AL4 + 1)
AL_d_inds = np.arange(AL5, AL6B + 1)

RL_s_inds = np.arange(RL1, RL4 + 1)
RL_d_inds = np.arange(RL5, RL6B + 1)

PM_s_inds = np.arange(PM1, PM4 + 1)
PM_d_inds = np.arange(PM5, PM6B + 1)

AM_s_inds = np.arange(AM1, AM4 + 1)
AM_d_inds = np.arange(AM5, AM6B + 1)


AdjMat = np.array(AdjMat)

#For getting a partricular element, e.g. V1_L4,V1_L4, need AdjMat[V1_L4, V1_L4 - 1]

Vis_AdjMat[0,:] = np.array([np.nan,                                                        np.sum((AdjMat[V1_L1:V1_L4 + 1, V1_L5 - 1: V1_L6B]).astype(float)),
                            np.sum((AdjMat[V1_L1:V1_L4 + 1, LM1 - 1: LM4]).astype(float)), np.sum((AdjMat[V1_L1:V1_L4 + 1, LM5 - 1: LM6B]).astype(float)),
                            np.sum((AdjMat[V1_L1:V1_L4 + 1, AL1 - 1: AL4]).astype(float)), np.sum((AdjMat[V1_L1:V1_L4 + 1, AL5 - 1: AL6B]).astype(float)),
                            np.sum((AdjMat[V1_L1:V1_L4 + 1, RL1 - 1: RL4]).astype(float)), np.sum((AdjMat[V1_L1:V1_L4 + 1, RL5 - 1: RL6B]).astype(float)),
                            np.sum((AdjMat[V1_L1:V1_L4 + 1, PM1 - 1: PM4]).astype(float)), np.sum((AdjMat[V1_L1:V1_L4 + 1, PM5 - 1: PM6B]).astype(float)),
                            np.sum((AdjMat[V1_L1:V1_L4 + 1, AM1 - 1: AM4]).astype(float)), np.sum((AdjMat[V1_L1:V1_L4 + 1, AM5 - 1: AM6B]).astype(float))])

Vis_AdjMat[1,:] = np.array([np.sum((AdjMat[V1_L5:V1_L6B + 1, V1_L1 - 1: V1_L4]).astype(float)), np.nan,
                            np.sum((AdjMat[V1_L5:V1_L6B + 1, LM1 - 1: LM4]).astype(float)), np.sum((AdjMat[V1_L5:V1_L6B + 1, LM5 - 1: LM6B]).astype(float)),
                            np.sum((AdjMat[V1_L5:V1_L6B + 1, AL1 - 1: AL4]).astype(float)), np.sum((AdjMat[V1_L5:V1_L6B + 1, AL5 - 1: AL6B]).astype(float)),
                            np.sum((AdjMat[V1_L5:V1_L6B + 1, RL1 - 1: RL4]).astype(float)), np.sum((AdjMat[V1_L5:V1_L6B + 1, RL5 - 1: RL6B]).astype(float)),
                            np.sum((AdjMat[V1_L5:V1_L6B + 1, PM1 - 1: PM4]).astype(float)), np.sum((AdjMat[V1_L5:V1_L6B + 1, PM5 - 1: PM6B]).astype(float)),
                            np.sum((AdjMat[V1_L5:V1_L6B + 1, AM1 - 1: AM4]).astype(float)), np.sum((AdjMat[V1_L5:V1_L6B + 1, AM5 - 1: AM6B]).astype(float))])

Vis_AdjMat[2,:] = np.array([np.sum((AdjMat[LM1:LM4 + 1, V1_L1 - 1: V1_L4]).astype(float)), np.sum((AdjMat[LM1:LM4 + 1, V1_L5 - 1: V1_L6B]).astype(float)),
                            np.nan,                                                        np.sum((AdjMat[LM1:LM4 + 1, LM5 - 1: LM6B]).astype(float)),
                            np.sum((AdjMat[LM1:LM4 + 1, AL1 - 1: AL4]).astype(float)),     np.sum((AdjMat[LM1:LM4 + 1, AL5 - 1: AL6B]).astype(float)),
                            np.sum((AdjMat[LM1:LM4 + 1, RL1 - 1: RL4]).astype(float)),     np.sum((AdjMat[LM1:LM4 + 1, RL5 - 1: RL6B]).astype(float)),
                            np.sum((AdjMat[LM1:LM4 + 1, PM1 - 1: PM4]).astype(float)),     np.sum((AdjMat[LM1:LM4 + 1, PM5 - 1: PM6B]).astype(float)),
                            np.sum((AdjMat[LM1:LM4 + 1, AM1 - 1: AM4]).astype(float)),     np.sum((AdjMat[LM1:LM4 + 1, AM5 - 1: AM6B]).astype(float))])

Vis_AdjMat[3,:] = np.array([np.sum((AdjMat[LM5:LM6B + 1, V1_L1 - 1: V1_L4]).astype(float)), np.sum((AdjMat[LM5:LM6B + 1, V1_L5 - 1: V1_L6B]).astype(float)),
                            np.sum((AdjMat[LM5:LM6B + 1, LM1 - 1: LM4]).astype(float)),     np.nan,
                            np.sum((AdjMat[LM5:LM6B + 1, AL1 - 1: AL4]).astype(float)),     np.sum((AdjMat[LM5:LM6B + 1, AL5 - 1: AL6B]).astype(float)),
                            np.sum((AdjMat[LM5:LM6B + 1, RL1 - 1: RL4]).astype(float)),     np.sum((AdjMat[LM5:LM6B + 1, RL5 - 1: RL6B]).astype(float)),
                            np.sum((AdjMat[LM5:LM6B + 1, PM1 - 1: PM4]).astype(float)),     np.sum((AdjMat[LM5:LM6B + 1, PM5 - 1: PM6B]).astype(float)),
                            np.sum((AdjMat[LM5:LM6B + 1, AM1 - 1: AM4]).astype(float)),     np.sum((AdjMat[LM5:LM6B + 1, AM5 - 1: AM6B]).astype(float))])

Vis_AdjMat[4,:] = np.array([np.sum((AdjMat[AL1:AL4 + 1, V1_L1 - 1: V1_L4]).astype(float)), np.sum((AdjMat[AL1:AL4 + 1, V1_L5 - 1: V1_L6B]).astype(float)),
                            np.sum((AdjMat[AL1:AL4 + 1, LM1 - 1: LM4]).astype(float)),     np.sum((AdjMat[AL1:AL4 + 1, LM5 - 1: LM6B]).astype(float)),
                            np.nan,                                                        np.sum((AdjMat[AL1:AL4 + 1, AL5 - 1: AL6B]).astype(float)),
                            np.sum((AdjMat[AL1:AL4 + 1, RL1 - 1: RL4]).astype(float)),     np.sum((AdjMat[AL1:AL4 + 1, RL5 - 1: RL6B]).astype(float)),
                            np.sum((AdjMat[AL1:AL4 + 1, PM1 - 1: PM4]).astype(float)),     np.sum((AdjMat[AL1:AL4 + 1, PM5 - 1: PM6B]).astype(float)),
                            np.sum((AdjMat[AL1:AL4 + 1, AM1 - 1: AM4]).astype(float)),     np.sum((AdjMat[AL1:AL4 + 1, AM5 - 1: AM6B]).astype(float))])

Vis_AdjMat[5,:] = np.array([np.sum((AdjMat[AL5:AL6B + 1, V1_L1 - 1: V1_L4]).astype(float)), np.sum((AdjMat[AL5:AL6B + 1, V1_L5 - 1: V1_L6B]).astype(float)),
                            np.sum((AdjMat[AL5:AL6B + 1, LM1 - 1: LM4]).astype(float)),     np.sum((AdjMat[AL5:AL6B + 1, LM5 - 1: LM6B]).astype(float)),
                            np.sum((AdjMat[AL5:AL6B + 1, AL1 - 1: AL4]).astype(float)),     np.nan,
                            np.sum((AdjMat[AL5:AL6B + 1, RL1 - 1: RL4]).astype(float)),     np.sum((AdjMat[AL5:AL6B + 1, RL5 - 1: RL6B]).astype(float)),
                            np.sum((AdjMat[AL5:AL6B + 1, PM1 - 1: PM4]).astype(float)),     np.sum((AdjMat[AL5:AL6B + 1, PM5 - 1: PM6B]).astype(float)),
                            np.sum((AdjMat[AL5:AL6B + 1, AM1 - 1: AM4]).astype(float)),     np.sum((AdjMat[AL5:AL6B + 1, AM5 - 1: AM6B]).astype(float))])

Vis_AdjMat[6,:] = np.array([np.sum((AdjMat[RL1:RL4 + 1, V1_L1 - 1: V1_L4]).astype(float)), np.sum((AdjMat[RL1:RL4 + 1, V1_L5 - 1: V1_L6B]).astype(float)),
                            np.sum((AdjMat[RL1:RL4 + 1, LM1 - 1: LM4]).astype(float)),     np.sum((AdjMat[RL1:RL4 + 1, LM5 - 1: LM6B]).astype(float)),
                            np.sum((AdjMat[RL1:RL4 + 1, AL1 - 1: AL4]).astype(float)),     np.sum((AdjMat[RL1:RL4 + 1, AL5 - 1: AL6B]).astype(float)),
                            np.nan,                                                        np.sum((AdjMat[RL1:RL4 + 1, RL5 - 1: RL6B]).astype(float)),
                            np.sum((AdjMat[RL1:RL4 + 1, PM1 - 1: PM4]).astype(float)),     np.sum((AdjMat[RL1:RL4 + 1, PM5 - 1: PM6B]).astype(float)),
                            np.sum((AdjMat[RL1:RL4 + 1, AM1 - 1: AM4]).astype(float)),     np.sum((AdjMat[RL1:RL4 + 1, AM5 - 1: AM6B]).astype(float))])


Vis_AdjMat[7,:] = np.array([np.sum((AdjMat[RL5:RL6B + 1, V1_L1 - 1: V1_L4]).astype(float)), np.sum((AdjMat[RL5:RL6B + 1, V1_L5 - 1: V1_L6B]).astype(float)),
                            np.sum((AdjMat[RL5:RL6B + 1, LM1 - 1: LM4]).astype(float)),     np.sum((AdjMat[RL5:RL6B + 1, LM5 - 1: LM6B]).astype(float)),
                            np.sum((AdjMat[RL5:RL6B + 1, AL1 - 1: AL4]).astype(float)),     np.sum((AdjMat[RL5:RL6B + 1, AL5 - 1: AL6B]).astype(float)),
                            np.sum((AdjMat[RL5:RL6B + 1, RL1 - 1: RL4]).astype(float)),     np.nan,
                            np.sum((AdjMat[RL5:RL6B + 1, PM1 - 1: PM4]).astype(float)),     np.sum((AdjMat[RL5:RL6B + 1, PM5 - 1: PM6B]).astype(float)),
                            np.sum((AdjMat[RL5:RL6B + 1, AM1 - 1: AM4]).astype(float)),     np.sum((AdjMat[RL5:RL6B + 1, AM5 - 1: AM6B]).astype(float))])

Vis_AdjMat[8,:] = np.array([np.sum((AdjMat[PM1:PM4 + 1, V1_L1 - 1: V1_L4]).astype(float)), np.sum((AdjMat[PM1:PM4 + 1, V1_L5 - 1: V1_L6B]).astype(float)),
                            np.sum((AdjMat[PM1:PM4 + 1, LM1 - 1: LM4]).astype(float)),     np.sum((AdjMat[PM1:PM4 + 1, LM5 - 1: LM6B]).astype(float)),
                            np.sum((AdjMat[PM1:PM4 + 1, AL1 - 1: AL4]).astype(float)),     np.sum((AdjMat[PM1:PM4 + 1, AL5 - 1: AL6B]).astype(float)),
                            np.sum((AdjMat[PM1:PM4 + 1, RL1 - 1: RL4]).astype(float)),     np.sum((AdjMat[PM1:PM4 + 1, RL5 - 1: RL6B]).astype(float)),
                            np.nan,                                                        np.sum((AdjMat[PM1:PM4 + 1, PM5 - 1: PM6B]).astype(float)),
                            np.sum((AdjMat[PM1:PM4 + 1, AM1 - 1: AM4]).astype(float)),     np.sum((AdjMat[PM1:PM4 + 1, AM5 - 1: AM6B]).astype(float))])

Vis_AdjMat[9,:] = np.array([np.sum((AdjMat[PM5:PM6B + 1, V1_L1 - 1: V1_L4]).astype(float)), np.sum((AdjMat[PM5:PM6B + 1, V1_L5 - 1: V1_L6B]).astype(float)),
                            np.sum((AdjMat[PM5:PM6B + 1, LM1 - 1: LM4]).astype(float)),     np.sum((AdjMat[PM5:PM6B + 1, LM5 - 1: LM6B]).astype(float)),
                            np.sum((AdjMat[PM5:PM6B + 1, AL1 - 1: AL4]).astype(float)),     np.sum((AdjMat[PM5:PM6B + 1, AL5 - 1: AL6B]).astype(float)),
                            np.sum((AdjMat[PM5:PM6B + 1, RL1 - 1: RL4]).astype(float)),     np.sum((AdjMat[PM5:PM6B + 1, RL5 - 1: RL6B]).astype(float)),
                            np.sum((AdjMat[PM5:PM6B + 1, PM1 - 1: PM4]).astype(float)),     np.nan,
                            np.sum((AdjMat[PM5:PM6B + 1, AM1 - 1: AM4]).astype(float)),     np.sum((AdjMat[PM5:PM6B + 1, AM5 - 1: AM6B]).astype(float))])

Vis_AdjMat[10,:] = np.array([np.sum((AdjMat[AM1:AM4 + 1, V1_L1 - 1: V1_L4]).astype(float)), np.sum((AdjMat[AM1:AM4 + 1, V1_L5 - 1: V1_L6B]).astype(float)),
                            np.sum((AdjMat[AM1:AM4 + 1, LM1 - 1: LM4]).astype(float)),      np.sum((AdjMat[AM1:AM4 + 1, LM5 - 1: LM6B]).astype(float)),
                            np.sum((AdjMat[AM1:AM4 + 1, AL1 - 1: AL4]).astype(float)),      np.sum((AdjMat[AM1:AM4 + 1, AL5 - 1: AL6B]).astype(float)),
                            np.sum((AdjMat[AM1:AM4 + 1, RL1 - 1: RL4]).astype(float)),      np.sum((AdjMat[AM1:AM4 + 1, RL5 - 1: RL6B]).astype(float)),
                            np.sum((AdjMat[AM1:AM4 + 1, PM1 - 1: PM4]).astype(float)),      np.sum((AdjMat[AM1:AM4 + 1, PM5 - 1: PM6B]).astype(float)),
                            np.nan,                                                         np.sum((AdjMat[AM1:AM4 + 1, AM5 - 1: AM6B]).astype(float))])

Vis_AdjMat[11,:] = np.array([np.sum((AdjMat[AM5:AM6B + 1, V1_L1 - 1: V1_L4]).astype(float)), np.sum((AdjMat[AM5:AM6B + 1, V1_L5 - 1: V1_L6B]).astype(float)),
                            np.sum((AdjMat[AM5:AM6B + 1, LM1 - 1: LM4]).astype(float)),     np.sum((AdjMat[AM5:AM6B + 1, LM5 - 1: LM6B]).astype(float)),
                            np.sum((AdjMat[AM5:AM6B + 1, AL1 - 1: AL4]).astype(float)),     np.sum((AdjMat[AM5:AM6B + 1, AL5 - 1: AL6B]).astype(float)),
                            np.sum((AdjMat[AM5:AM6B + 1, RL1 - 1: RL4]).astype(float)),     np.sum((AdjMat[AM5:AM6B + 1, RL5 - 1: RL6B]).astype(float)),
                            np.sum((AdjMat[AM5:AM6B + 1, PM1 - 1: PM4]).astype(float)),     np.sum((AdjMat[AM5:AM6B + 1, PM5 - 1: PM6B]).astype(float)),
                            np.sum((AdjMat[AM5:AM6B + 1, AM1 - 1: AM4]).astype(float)),     np.nan])


Vis_AdjMat[0:2, 0:2] = np.nan
Vis_AdjMat[2:4, 2:4] = np.nan
Vis_AdjMat[4:6, 4:6] = np.nan
Vis_AdjMat[6:8, 6:8] = np.nan
Vis_AdjMat[8:10, 8:10] = np.nan
Vis_AdjMat[10:12, 10:12] = np.nan

# Normalize by roe
# row_max = np.nanmax(Vis_AdjMat, axis=1)
# Vis_AdjMat = Vis_AdjMat / row_max[:, np.newaxis]

out_degrees = np.nansum(Vis_AdjMat, axis = 1)
in_degrees  = np.nansum(Vis_AdjMat, axis = 0)
ratio = in_degrees/out_degrees

fig = plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.imshow(Vis_AdjMat, cmap = 'magma')
plt.xticks(np.arange(len(Area_order)), Area_order, rotation =90)
plt.yticks(np.arange(len(Area_order)), Area_order)
plt.xlabel('Target')
plt.ylabel('Source')
plt.subplot(1,2,2)

plt.scatter(np.arange(len(Area_order)), ratio, lw = 10)
plt.xticks(np.arange(len(Area_order)), Area_order, rotation = 90)
plt.ylabel('in-degree/out-degree')
plt.tight_layout()
plt.savefig("adjacency_matrix/anatomical_12x12_degree_quantification.pdf", format = 'pdf', dpi = 1000)

plt.show()