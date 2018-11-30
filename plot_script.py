import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import sys
import glob

colors_map = {
    'semi-commnet': '#dc8d6d',
    'commnet': '#5785c1',
    'semi-mlp': '#78b38a',
    'mlp': '#ba71af'
}

def read_file(vec, file_name, scalar, term):
    print(file_name)
    with open(file_name, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            return vec

        mean_reward = False
        for idx, line in enumerate(lines):
            if term not in line:
                continue
            epoch_idx = idx
            epoch_line = line
            while 'Epoch' not in epoch_line:
                epoch_idx -= 1
                epoch_line = lines[epoch_idx]

            epoch = int(epoch_line.split(' ')[1].split('\t')[0])
            if not scalar:
                floats = line.split('\t')[1]
                left_bracket = floats.find('[')
                right_bracket = floats.find(']')
                floats = np.fromstring(floats[left_bracket + 2:right_bracket], dtype=float, sep=' ')

                if epoch > len(vec):
                    vec.append([floats.mean()])
                else:
                    vec[epoch - 1].append(floats.mean())
            else:
                floats = line.split('\t')[0]
                if epoch > len(vec):
                    vec.append([float(floats.split(' ')[-1].strip())])
                else:
                    vec[epoch - 1].append(float(floats.split(' ')[-1].strip()))

                # if term == 'Mean':
                #     vec.append(float(floats.split(' ')[2]))
                # elif term == 'Success':
                #     vec.append(float(floats.split(' ')[1].strip()))
                # else:
                #     raise RuntimeError("Unkown term used as line parser.")

    return vec

number = 1

def parse_plot(files, scalar=False, term='Epoch'):
    global number
    fig = plt.subplot(6, 1, number)
    number += 2
    coll = dict()
    for fname in files:
        f = fname.split('.')
        if 'semi' in fname and 'commnet' in fname:
            label = 'semi-commnet'
        elif 'commnet' in fname:
            label = 'commnet'
        elif 'semi' in fname and 'mlp' in fname:
            label = 'semi-mlp'
        else:
            label = 'mlp'

        if label not in coll:
            coll[label] = []

        # if len(f) == 2:
        #     label = f[0]
        # else:
        #     label = '-'.join(f[0].split('_')[1:])
        coll[label] = read_file(coll[label], fname, scalar, term)

    for label in coll.keys():
        coll[label] = coll[label][:1000]

        mean_values = []
        max_values = []
        min_values = []

        for val in coll[label]:
            mean = sum(val) / len(val)

            if term == 'Success':
                mean *= 100
            mean_values.append(mean)
            variance = np.var(val)

            if term == 'Success':
                variance *= 100
            variance = variance if variance < 20 else 20
            max_values.append(mean + variance)
            min_values.append(mean - variance)

        fig.plot(np.arange(len(coll[label])), mean_values, linewidth=1.5, label=label, color=colors_map[label])
        fig.fill_between(np.arange(len(coll[label])), min_values, max_values, color=colors.to_rgba(colors_map[label], alpha=0.2))

    term = 'Rewards' if term == 'Epoch' else term

    fig.set_xlabel('Epochs')
    fig.set_ylabel(term)
    fig.legend()
    fig.grid()
    fig.set_title('StarCraft {} {}'.format(sys.argv[2], term))

files = glob.glob(sys.argv[1] + "*")
files = list(filter(lambda x: x.find(".pt") == -1, files))

# rewards
parse_plot(files, False, 'Epoch')

# success
parse_plot(files, True, 'Success')

# steps
parse_plot(files, True, 'Steps taken')
plt.show()
