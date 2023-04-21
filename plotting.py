import numpy as np
import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import difflib
import pandas as pd
import datetime
import matplotlib.font_manager
import matplotlib as mpl
from matplotlib.font_manager import FontProperties #字体管理器
'''
simhei.ttf
simsun.ttc
'''

eval_item = ['episode', 'episode_reward', 'eval_time', 'mean_episode_reward', 'best_episode_reward', 'step']
train_item = ["episode_reward", "episode", "batch_reward", "critic_loss", "actor_loss", "actor_target_entropy",
              "actor_entropy", "alpha_loss", "alpha_value", "curl_loss", "states", "action_state", "duration", "step"]

def read_log(alg, file_name, domain_task, mode='eval'):
    '''
    Read json dict data in log file
    input:
        file_name: csc_file_name [str]
        action_repeat: [int]
    outputs:
        data: [dict]
    '''
    if alg == 'dreamer':
        xaxis = 'step'
        yaxis = 'test/return'
        with open(file_name) as f:
            df = pd.DataFrame([json.loads(l) for l in f.readlines()])
            df = df[[xaxis, yaxis]].dropna()
            xs = df[xaxis].to_numpy()
            ys = df[yaxis].to_numpy()
            #xs, ys = binning(xs, ys, bins, np.nanmean)
            # xs = xs[0:51]
            # xs[50] = 500000
            # if domain_task == "cheetah_run":
            #     xs = xs[0:201]
            #     ys = ys[0:201]
            #     xs[200] = 2000000
            #     ys[200] = (ys[199] + ys[200]) / 2
            # # elif domain_task == "reacher_easy":
            # #     xs = xs[0:101]
            # #     ys = ys[0:101]
            # #     xs[100] = 1000000
            # #     ys[100] = (ys[99] + ys[100]) / 2
            # else:
            #     xs = xs[0:51]
            #     xs[50] = 500000
            #     ys = ys[0:51]
            #     ys[50] = (ys[49] + ys[50])/2
            return xs, ys, (ys[9]+ys[10])/2, ys[50]
    else:
        assert mode=='eval' or mode=='train', 'Mode should choose eval or train'
        if mode== 'eval':
            item_list = eval_item
        else:
            item_list = train_item
        init_value = [[] for i in range(len(item_list))]
        data = dict(zip(item_list, init_value))
        with open(file_name) as f:
            for json_data in f:
                element = json.loads(json_data)
                if len(element) == len(item_list): # handle eval.log
                    for item in item_list:
                        data[item].append(element[item])
                else: # handle train.log
                    for item in item_list:
                        if item in element:
                            data[item].append(element[item])

                if domain_task == "cheetah_run":
                    if element['episode'] == 2000:
                        break
                # elif domain_task == "reacher_easy":
                #     if element['episode'] == 1000:
                #         break
                else:
                    if element['episode'] == 500:
                        break
        if mode=='train':
            # eliminate repeated step data when evaluating
            data['step'] = list(set(data['step']))

        data['environment_step'] = [value * 1000 for value in data['episode']]
        if 'episode' in data :
            if 100 in data['episode']:
                index = data['episode'].index(100)
                reward_100 = data['mean_episode_reward'][index]
            else:
                reward_100 = None
            if 500 in data['episode']:
                index = data['episode'].index(500)
                reward_500 = data['mean_episode_reward'][index]
            else:
                reward_500 = None

    return data, reward_100, reward_500


num_seed = 5

plot_type = ['ablation', 'efficiency', 'gener']
# read logs
domain_name = ['walker']
task_name = ['run']
algs = ['Dreamer']
init_value = [[] for i in range(len(domain_name))]

cm = 1/2.54
font_size = 10

### Set grid
mpl.rcParams['font.family'] = 'Arial'
sns.set_style("whitegrid", {'grid.linestyle': '--'})
### Set font size in text, title, labelsize
sns.set_context("paper", rc={"font.size": font_size, "axes.titlesize": font_size, "axes.labelsize": font_size, "font.name":"Arial"})
fig, axes = plt.subplots(1, 1)
fig.set_size_inches(10*cm, 8*cm)
for index, (domain, task) in enumerate(zip(domain_name, task_name)):

    for alg in algs:
        linestyle = '-'
        predir = './logdir/dmc_{}_{}'.format(domain, task)
    
        if alg == 'Dreamer':
            color = 'r'
            condition = 'Dreamer'
        if alg == 'img_prop':
            color = 'g'
            condition = 'img_prop'

        config_reward = []
        config_step = []

        for seed in range(num_seed):
            if alg == "Dreamer":
                file_dir = predir + '/' + 'dreamer-s{}'.format(str(seed)) + '/' + 'metrics.jsonl'
                steps, reward, reward100, reward500 = read_log('dreamer', file_dir,
                                                               domain_task=str(domain) + '_' + str(task), mode='eval')

                config_step.append(steps)
                config_reward.append(reward)
            else:
                dir = predir + '/' + domain + '-' + task + '-s' + str(seed)
                file_name = 'eval.log'
                file_dir = dir + '/' + file_name
                args_file_dir = dir + '/' + 'args.json'

                data, reward100, reward500 = read_log(alg, file_dir, domain_task= str(domain) + '_' + str(task), mode='eval')

                config_step.append(data['environment_step'])
                config_reward.append(data['mean_episode_reward'])
        min_length = len(min(config_step, key= len))
        config_step = config_step[0][:min_length]
        config_reward = [reward[:min_length] for reward in config_reward]

        ax_x, ax_y =  index//2, index % 2

        ax = axes

        g = sns.tsplot(ax=ax, time = config_step, data= config_reward, color=color, linestyle=linestyle, condition=condition, ci=95)

        if alg == "MIDY":
            zorder = 10
            plt.setp(g.lines, zorder=zorder)

    if domain == 'cheetah':
        g.set(xlim=(-60000, 2060000))

        xlabels = ['-0.5', '0.0', '0.5', '1.0', '1.5', '2.0', '2.5']

    else:
        g.set(xlim=(-15000, 515000))

        xlabels = ['-0.1', '0.0', '0.1', '0.2', '0.3', '0.4', '0.5']
    ax.set_xticklabels(xlabels)

    if domain == "cheetah" or domain == "cartpole":
        ylabels = [0, 200, 400, 600, 800]
    else:
        ylabels = [0, 200, 400, 600, 800, 1000]

    ax.set_yticks(ylabels)

    ax.get_legend().remove()

    ### Set ticks attributes
    ax.tick_params(axis='both', direction='in', length=2, pad=1.3, labelsize=font_size - 1, bottom=True, left=True)

    ax.set_ylabel("reward", labelpad=0.6)
    ax.set_xlabel("Environment Steps(x$10^6$)",labelpad=0.6)
    if domain == 'ball_in_cup':
        domain ='ball-in-cup'
    ax.set_title(domain.capitalize() +' '+ task.capitalize())

    handles, labels = ax.get_legend_handles_labels()

fig.tight_layout()
fig.subplots_adjust(bottom=0.16)  # create some space below the plots by increasing the bottom-value
fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.95, 0.15), ncol=1, fontsize=8)

plt.show()

outdir = './plot'
dpi = 300
time = datetime.datetime.now()
time = datetime.datetime.strptime(str(time), '%Y-%m-%d %H:%M:%S.%f')
filename = 'ablation_ci_95' + "_dpi" + str(dpi) + '_' + str(time.date())+'.png'
filename = outdir + '/' + filename
fig = g.get_figure()
fig.savefig(filename, dpi=dpi)