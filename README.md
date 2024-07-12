# MTE_PartialFrequency_CSILocalization

The Code for Paper "3D Indoor Localization Using Partial-Frequency CSI Fingerprints with Mask Transformer Encoder"

Please run MTE.py, change sy/xh for two scenarios.:
1) scenario_name = 'sy_all';# sy_all[4, 73, 25, 15, 2, 2048] w=[0.25,0.25,0.5], xh_all [4, 81, 25, 13, 2, 2048] w=[0.275, 0.25, 0.25]
2) DATA_PATH = 'sy_data.pt' # 'sy_data.pt' 'xh_data.pt' 

Set Mask Ratio if needed:
NUM_MUSK = 0; # Set how many random sub-channels unavailable 0-8

Data is avaiblable at: https://drive.google.com/drive/folders/1cNlbyR5yhDjr-fpA2fttO7pcKPuFp-5m?usp=sharing
