import os
import torch
import numpy as np
import pandas as pd
from environment import Environment
from tensorboardX import SummaryWriter
from model import Dueling_Q_Network
from utils import policy, shuffle_tensor
from finta import TA
from plotly.graph_objs import *
import streamlit as st


def rl_agent_test(model, env, step_max, epsilon, global_step, rewards, sharpe, num_epoch, plot_data, init_money):
    for epoch in range(num_epoch): 
        state = env.reset()
        step = 0
        total_reward = 0
        total_sharpe = []
        sharpe = 0
        plot_data['buy'] = 0
        plot_data['sell'] = 0

        while True and step < step_max:
            action = policy(model=model, state=state, epsilon=epsilon)
            next_state, reward, record_reward = env(action, plot_data)
            total_reward += record_reward
            total_sharpe.append(reward)
            state = next_state
            step += 1
            global_step += 1

        mean_sharpe_list = [i for i in total_sharpe if i != 0]
        mean_sharpe = sum(mean_sharpe_list) / len(mean_sharpe_list)
        rewards.append(total_reward)
    return rewards, mean_sharpe


def test(env, model, plot_data, init_money):

    global_step = 0
    rewards = []
    sharpe = []
    losses = []

    rewards, mean_sharpe = rl_agent_test(
        model=model,
        env=env,
        step_max=len(env.data)-1,
        global_step=global_step,
        rewards=rewards,
        sharpe=sharpe,
        num_epoch=1,
        epsilon=1.0,
        plot_data=plot_data,
        init_money=init_money)

    return model, rewards, mean_sharpe


@st.cache(suppress_st_warning=True)
def plot(plot_data, rewards, mean_sharpe):
    trace1 = {
        "line": {
            "dash": "solid",
            "color": "#0E75FF",
            "width": 2
        },
        "mode": "lines",
        "type": "scatter",
        "x": plot_data['date'],
        "y": plot_data['close'],
        "yaxis": "y",
        "text": plot_data['close'].values
    }
    trace2 = {
        "mode": "markers",
        "name": "Buy",
        "type": "scatter",
        "marker_symbol": "triangle-up",
        "x": plot_data.loc[plot_data['buy'] == 1, 'date'].values,
        "y": plot_data.loc[plot_data['buy'] == 1, 'close'].values,
        "yaxis": "y",
        "text": plot_data.loc[plot_data['buy'] == 1, 'close'].values,
        "marker": {
            "size": 12,
            "color": "#0EFF35",
            "symbol": 200
        }
    }
    trace3 = {
        "mode": "markers",
        "name": "Sell",
        "type": "scatter",
        "marker_symbol": "triangle-down",
        "x": plot_data.loc[plot_data['sell'] == 1, 'date'].values,
        "y": plot_data.loc[plot_data['sell'] == 1, 'close'].values,
        "yaxis": "y",
        "text": plot_data.loc[plot_data['sell'] == 1, 'close'].values,
        "marker": {
            "size": 12,
            "color": "red",
            "symbol": 200
        }
    }

    data = Data([trace1, trace2, trace3])
    fig = Figure(data=data)

    st.write("""
    # Testing Result
    (from 2020/08/01 to 2021/08/01)
    """)

    st.write("##### 總報酬：{:.4f}".format(rewards[0]))
    st.write("##### 夏普值：{:.4f}".format(mean_sharpe))
    st.plotly_chart(fig, use_container_width=True)


def read_data(data, start_date, split_date):

    data = data
    data['Date'] = pd.to_datetime(data['Date'])

    lowcols = []
    for lowcol in data:
        lowcols.append(lowcol.lower())
    data.columns = lowcols

    macd = TA.MACD(data)
    s_vama = TA.VAMA(data, period=10)  
    l_vama = TA.VAMA(data, period=20)  
    data = pd.concat([data, macd, s_vama, l_vama],
                     axis=1)  

    data = data.set_index('date')
    data = data.sort_values("date")

    if start_date:
        train_data = data[start_date:split_date]
    else:
        train_data = data[:split_date]
    test_data = data[split_date:]

    return train_data, test_data


def main(data, param_file):
    train_data, test_data = read_data(
        data=data,
        start_date='2016-05-18',
        split_date='2020-08-01')

    init_money = 100000

    test_env = Environment(
        data=test_data, history_length=30, init_money=init_money)

    model = Dueling_Q_Network(30)

    model_dict = model.state_dict()
    checkpoint = torch.load(param_file)['state_dict']
    checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(checkpoint)
    model.load_state_dict(model_dict)

    plot_data = test_data.reset_index()

    model, rewards, mean_sharpe = test(
        env=test_env,
        model=model,
        plot_data=plot_data,
        init_money=init_money)

    plot(plot_data, rewards, mean_sharpe)
    return


st.set_page_config(page_title="ETF Testing Result", layout="wide")

st.write('''
# DQN Testing Web APP
**請上傳 ETF 資料(csv 檔)及參數(pth 檔)**
''')

st.sidebar.header("上傳您的檔案")

with st.sidebar.subheader("1. ETF Data(csv)"):
    etf_file = st.sidebar.file_uploader("上傳您的 ETF 資料", type=['csv'])

with st.sidebar.subheader("2. Parameters(pth)"):
    param_file = st.sidebar.file_uploader("上傳您的參數檔案", type=['pth'])

if (etf_file is not None) and (param_file is not None):
    st.subheader("Your Data")
    data = pd.read_csv(etf_file)
    st.write(data)
    main(data, param_file)
else:
    st.write("Waiting for upload the data...")
    if st.button("Use example data"):
        st.subheader("Your Data")
        data = pd.read_csv("https://raw.githubusercontent.com/Alan0329/Alan0329/main/Other/DQN_Testing_Web_APP/00646.TW.csv")
        st.write(data)
        param_file = "https://github.com/Alan0329/Alan0329/raw/main/Other/DQN_Testing_Web_APP/200_checkpoint.pth"
        main(data, param_file)
