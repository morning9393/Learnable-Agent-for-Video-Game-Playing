import queue
import threading
import tkinter as tk
import numpy as np
import tensorflow as tf
from game_wrapper import make_atari, wrap_deepmind
from SA2C import Model, A2C

GAME = 'Game:'
PRE_TRAINED = 'Pre-trained weights:'
LR = 'Learning Rate:'
GAMMA = 'Reward Discount Factor:'
BATCH = 'Batch Size:'
EPISODE = 'Number of Episode:'
COEFFICIENT = 'Coefficient of Entropy:'
LOG = 'Log:'
BREAKOUT = 'BreakoutNoFrameskip-v4'
SPACE_INVADERS = 'SpaceInvadersNoFrameskip-v4'
BRE_15000 = './pre-trained_weights/Breakout/15000/'
BRE_30000 = './pre-trained_weights/Breakout/30000/'
SPA_15000 = './pre-trained_weights/Space_Invaders/15000/'
SPA_30000 = './pre-trained_weights/Space_Invaders/30000/'

a2c = None
env = None


def train():
    game = game_var.get()
    pre = pre_var.get()
    lr = float(lr_var.get())
    gamma = float(gamma_var.get())
    batch = int(batch_var.get())
    episode = int(episode_var.get())
    coe = float(coe_var.get())

    global env
    pre_path = None
    if game:
        env = make_atari(SPACE_INVADERS)
        if pre == 1:
            pre_path = SPA_15000
        if pre == 2:
            pre_path = SPA_30000
    else:
        if pre == 1:
            pre_path = BRE_15000
        if pre == 2:
            pre_path = BRE_30000
        env = make_atari(BREAKOUT)
    env = wrap_deepmind(env, clip_rewards=False, frame_stack=True, scale=True)
    env.seed(seed)

    model = Model(3)
    global a2c
    a2c = A2C(model, env, lr, episode, gamma, batch, coe, pre_weight_path=pre_path, log=log_text, queue=msg_queue)
    a2c.train()


def train_clicked():
    train_button.config(state=tk.DISABLED)
    th = threading.Thread(target=train)
    th.setDaemon(True)
    th.start()
    # train()


def stop_clicked():
    a2c.stop()
    global msg_queue
    msg_queue = queue.Queue()
    train_button.config(state=tk.ACTIVE)


def show():
    while not msg_queue.empty():
        msg_queue.get()
        env.render()
    root.after(50, show)


seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

root = tk.Tk()
root.title('SA2C_Demo')
root.geometry('800x600')

control_frame = tk.Frame(root)

param_frame = tk.Frame(control_frame)

game_frame = tk.Frame(param_frame)
game_frame.grid(row=0, column=0, sticky='w')
game_label = tk.Label(game_frame, text=GAME)
game_label.pack(side=tk.LEFT)
game_var = tk.IntVar()
game_Space = tk.Radiobutton(game_frame, text='Space Invaders', variable=game_var, value=1)
game_Space.pack(side=tk.RIGHT)
game_breakout = tk.Radiobutton(game_frame, text='Breakout', variable=game_var, value=0)
game_breakout.pack(side=tk.RIGHT)
# game_frame.pack(side=tk.TOP)

pre_frame = tk.Frame(param_frame)
pre_frame.grid(row=1, column=0, sticky='w')
pre_label = tk.Label(pre_frame, text=PRE_TRAINED)
pre_label.pack(side=tk.LEFT)
pre_var = tk.IntVar()
pre_30000 = tk.Radiobutton(pre_frame, text='From 30000 episodes', variable=pre_var, value=2)
pre_30000.pack(side=tk.RIGHT)
pre_15000 = tk.Radiobutton(pre_frame, text='From 15000 episodes', variable=pre_var, value=1)
pre_15000.pack(side=tk.RIGHT)
pre_none = tk.Radiobutton(pre_frame, text='None', variable=pre_var, value=0)
pre_none.pack(side=tk.RIGHT)
# pre_frame.pack(side=tk.TOP)

lr_frame = tk.Frame(param_frame)
lr_frame.grid(row=2, column=0, sticky='w')
lr_label = tk.Label(lr_frame, text=LR)
lr_label.pack(side=tk.LEFT)
lr_var = tk.StringVar()
lr_entry = tk.Entry(lr_frame, textvariable=lr_var)
lr_var.set(0.0007)
lr_entry.pack(side=tk.RIGHT)
# lr_frame.pack(side=tk.TOP)

gamma_frame = tk.Frame(param_frame)
gamma_frame.grid(row=3, column=0, sticky='w')
gamma_label = tk.Label(gamma_frame, text=GAMMA)
gamma_label.pack(side=tk.LEFT)
gamma_var = tk.StringVar()
gamma_entry = tk.Entry(gamma_frame, textvariable=gamma_var)
gamma_var.set(0.99)
gamma_entry.pack(side=tk.RIGHT)
# gamma_frame.pack(side=tk.TOP)

batch_frame = tk.Frame(param_frame)
batch_frame.grid(row=4, column=0, sticky='w')
batch_label = tk.Label(batch_frame, text=BATCH)
batch_label.pack(side=tk.LEFT)
batch_var = tk.StringVar()
batch_entry = tk.Entry(batch_frame, textvariable=batch_var)
batch_var.set(100)
batch_entry.pack(side=tk.RIGHT)
# batch_frame.pack(side=tk.TOP)

episode_frame = tk.Frame(param_frame)
episode_frame.grid(row=5, column=0, sticky='w')
episode_label = tk.Label(episode_frame, text=EPISODE)
episode_label.pack(side=tk.LEFT)
episode_var = tk.StringVar()
episode_entry = tk.Entry(episode_frame, textvariable=episode_var)
episode_var.set(30000)
episode_entry.pack(side=tk.RIGHT)
# episode_frame.pack(side=tk.TOP)

coe_frame = tk.Frame(param_frame)
coe_frame.grid(row=6, column=0, sticky='w')
coe_label = tk.Label(coe_frame, text=COEFFICIENT)
coe_label.pack(side=tk.LEFT)
coe_var = tk.StringVar()
coe_entry = tk.Entry(coe_frame, textvariable=coe_var)
coe_var.set(0.01)
coe_entry.pack(side=tk.RIGHT)
# coe_frame.pack(side=tk.TOP)

param_frame.pack(side=tk.LEFT)

button_frame = tk.Frame(control_frame)

train_button = tk.Button(button_frame, text='Train', command=train_clicked)
train_button.pack()
stop_button = tk.Button(button_frame, text='Stop', command=stop_clicked)
stop_button.pack()

button_frame.pack(side=tk.RIGHT)

control_frame.pack(side=tk.TOP)

log_frame = tk.Frame(root)

log_label = tk.Label(log_frame, text=LOG)
log_label.pack(side=tk.LEFT)
log_text = tk.Text(log_frame, borderwidth=3, relief="sunken")
log_text.pack()

log_frame.pack(side=tk.TOP)

msg_queue = queue.Queue()
root.after(100, show)
root.mainloop()
