import pandas as pd


def preprocessing(annotation, duration=60):
    """
    Get annotation file format .edf and return table with signals in same duration
    :param duration: duration for signals (samples of signal)
    :param annotation: file with annotation
    :return: table with signals in same duration
    """
    sleep_stages = pd.DataFrame(columns=['onset', 'duration', 'stage'])
    for x in annotation:
        sleep_stages = sleep_stages.append(
            {'onset': x['onset'], 'duration': x['duration'], 'stage': x['description'][-1:]}, ignore_index=True)
    sleep_stages = sleep_stages.loc[sleep_stages['stage'].isin(['1', '2', '3', '4', 'W', 'R'])]
    for index, sleep in sleep_stages.iterrows():
        if sleep['duration'] > 60.0:
            for i in range(int(sleep['duration'] // duration)):
                sleep_stages = sleep_stages.append(
                    {'onset': sleep['onset'] + i * duration, 'duration': duration, 'stage': sleep['stage']},
                    ignore_index=True)
    sleep_stages = sleep_stages.loc[sleep_stages['duration'] == 60]
    return sleep_stages


def table_with_signals(sleep_stages, raw):
    """
    Return the table with signals and sleep-stage
    :param sleep_stages: table with sleep stages start point and duration for each stage
    :param raw: raw signal
    :return: table with sample of signal and its stage
    """
    signals = list()
    for i, s in sleep_stages.iterrows():
        signals.append(raw[:, int(s['onset']):int(s['onset']) + int(s['duration'])][0])

    eeg_signals = list()
    for i, s in enumerate(signals):
        d = dict()
        d['stage'] = sleep_stages.iloc[i]['stage']
        d['signal'] = s
        eeg_signals.append(d)

    eeg_signals = pd.DataFrame(eeg_signals)
    return eeg_signals
