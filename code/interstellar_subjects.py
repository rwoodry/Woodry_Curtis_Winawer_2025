'''
Dictionary of wlsubj parameters to source when running pipeline.
'''

wlsubjects = {
    'wlsubj139': {
        'wlsubj': 139,
        'sessionindicator': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        'sessions':['nyu3t01'],
        'sacc_dur': [1, 1.5],
        'n_TR': [280, 315],
        'design_filenames':
        [
            #Session 1
            'sub-wlsubj139_run-01_perception_07-16-22_14-25_trialdesign.tsv',
            'sub-wlsubj139_run-02_ltm_07-16-22_14-33_trialdesign.tsv',
            'sub-wlsubj139_run-03_wm_07-16-22_14-39_trialdesign.tsv',
            'sub-wlsubj139_run-04_perception_07-16-22_14-45_trialdesign.tsv',
            'sub-wlsubj139_run-05_ltm_07-16-22_14-50_trialdesign.tsv',
            'sub-wlsubj139_run-06_wm_07-16-22_14-56_trialdesign.tsv',
            
            #Session 2
            'sub-wlsubj139_run-07_perception_08-04-22_18-25_trialdesign.tsv',
            'sub-wlsubj139_run-08_ltm_08-04-22_18-31_trialdesign.tsv',
            'sub-wlsubj139_run-09_wm_08-04-22_18-37_trialdesign.tsv',
            'sub-wlsubj139_run-10_perception_08-04-22_18-44_trialdesign.tsv',
            'sub-wlsubj139_run-11_ltm_08-04-22_18-50_trialdesign.tsv',
            'sub-wlsubj139_run-12_wm_08-04-22_18-56_trialdesign.tsv'  
        ],
        'exclude_trials': [15, 31, 47, 63, 79, 95, 129, 146, 180, 197]
    },

    'wlsubj136': {
        'wlsubj': 136,
        'sessionindicator': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        'sessions':['nyu3t01', 'nyu3t02'],
        'sacc_dur': 1.5,
        'n_TR': 296,
        'exclude_trials': []
    },

    'wlsubj114': {
        'wlsubj': 114,
        'sessionindicator': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        'sessions':['nyu3t01', 'nyu3t02'],
        'sacc_dur': 1.5,
        'n_TR': 296,
        'exclude_trials': [],
    },
    
    'wlsubj127': {
        'wlsubj': 127,
        'sessionindicator': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        'sessions':['nyu3t01', 'nyu3t02'],
        'sacc_dur': 1.5,
        'n_TR': 296,
        'exclude_trials': [],
    },

    'wlsubj115': {
        'wlsubj': 115,
        'sessionindicator': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        'sessions':['nyu3t01', 'nyu3t02'],
        'sacc_dur': 1.5,
        'n_TR': 296,
        'exclude_trials': [],
    },

    'wlsubj135': {
        'wlsubj': 135,
        'sessionindicator': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        'sessions':['nyu3t01', 'nyu3t02'],
        'sacc_dur': 1.5,
        'n_TR': 296,
        'exclude_trials': [],
        'exclude_designs':
            	 ['sub-wlsubj135_run-01_perception_11-10-22_11-10_trialdata.tsv',
                 'sub-wlsubj135_run-01_perception_11-10-22_11-24_trialdata.tsv']
    },

    'wlsubj142': {
        'wlsubj': 142,
        'sessionindicator': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        'sessions':['nyu3t01', 'nyu3t02'],
        'sacc_dur': 1.5,
        'n_TR': 296,
        'exclude_trials': [],
    },

    'wlsubj141': {
        'wlsubj': 142,
        'sessionindicator': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        'sessions':['nyu3t01', 'nyu3t02'],
        'sacc_dur': 1.5,
        'n_TR': 296,
        'exclude_trials': [],
    }
}
