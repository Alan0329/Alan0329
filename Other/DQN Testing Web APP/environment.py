import numpy as np


class Environment():
    def __init__(self, data, history_length=90, init_money=100000):
        self.data = data
        self.history_length = history_length  # ?
        self.init_money = init_money  # 投入金額
        self.reset()  # reset()：讓環境在一開始,結束的時候重置環境

    def reset(self):
        self.t = 0  # 時間點0
        self.profits = 0
        self.account_money = self.init_money  # 投入金額
        self.buy_size = []  # 持有數
        self.hold_value = []  # 持有金額
        self.positions = []  # 持有數(放收盤價)
        self.position_value = 0  # 持有股票金額
        self.hold_day = []
        # history_length長度的0
        self.history = [[0, 0, 0, 0, 0, 0]
                        for _ in range(self.history_length)]
        return self.history  # state = 0

    def __call__(self, action, plot_data):
        reward = 0
        sharpe = 0
        record_reward = 0
        profits = 0
        hold_amount = 0
        daily_data = []
        handling_fee = 0.001425  # 手續費
        tax = 0.001  # 交易稅
        manage_fee = 0.0043  # 管理費用
        risk_free_rates = 0.004275

        # 0:stay, 1:buy, 2:sell
        # 若動作為買 ，且帳戶餘額在買進[價格*股數*(1+手續費)]後 > 0，則買進
        if action == 1 and self.account_money > self.data.iloc[self.t, :]["close"]:
           # 可買的股數
            buy_size = self.account_money // self.data.iloc[self.t, :]["close"]
            self.buy_size.append(buy_size)  # 將持有數記下來，將來賣掉計算用
            # 挑時間點t的收盤價，將來賣掉計算收益用
            self.positions.append(self.data.iloc[self.t, :]["close"])
            # 股票持有金額 = 收盤價 * 買進的股數
            self.hold_value.append(
                self.data.iloc[self.t, :]["close"] * buy_size)
            self.hold_day.append(self.t)  # 紀錄何時買進

            # 帳戶餘額 = 目前帳戶持有金額 - [目前收盤價 * 買進股數 * (1 + 買進手續費)]
            self.account_money -= (self.data.iloc[self.t, :]
                                   ["close"] * buy_size * (1 + handling_fee))
            # 紀錄買進價格、股數、目前帳戶餘額
            print("以 %.5f的價格，買進 %6d股：目前帳戶%6d 元" % (
                self.data.iloc[self.t, :]["close"], buy_size, self.account_money))
            # 繪圖用
            plot_data.iloc[self.t, 10] = 1
        elif action == 2:  # 賣
            if len(self.buy_size) == 0:
                reward_signal = -1
            else:
                if ((self.t - self.hold_day[-1]) < 2) or (self.data.iloc[self.t, :]['close'].item() in self.data.iloc[self.t-3:self.t, :]['close'].tolist()):
                    reward_signal = -1
                else:
                    # 遍歷持有的股票價格
                    for i in range(len(self.positions)):
                        # 收益 = (目前股價 - 之前購買股價) * 之前購買股數
                        # -> 舉例：第一次買100元、2股；第二次買150元、10股，現在賣出300元
                        # 收益 = (300-100) * 2 + (300-150) * 10 = 1900
                        profits += ((self.data.iloc[self.t, :]["close"] -
                                    self.positions[i]) * self.buy_size[i])
                        # 將每次賣出後得到價錢加到帳戶
                        self.account_money += self.data.iloc[self.t,
                                                             :]["close"] * self.buy_size[i]

                    # 收益 = 收益 - (目前價格*賣出股數)*(賣出手續費+交易稅)
                    profits -= self.data.iloc[self.t, :]["close"] * \
                        sum(self.buy_size) * (handling_fee + tax)
                    # 同上
                    self.account_money = self.account_money - \
                        self.data.iloc[self.t, :]["close"] * \
                        sum(self.buy_size) * (handling_fee + tax)
                    # 紀錄賣出價格、損益、帳戶餘額
                    print("以 %.5f元賣出，此次投資損益： %6d 元：目前帳戶%6d 元" % (
                        self.data.iloc[self.t, :]["close"], profits, self.account_money))
                    reward = profits / sum(self.hold_value)  # 收益

                    sharpe_ratio_list = []  # 用來保存夏普值
                    # 遍歷買進天數
                    for i in self.hold_day:
                        # 計算平均報酬
                        r_data = self.data.copy(deep=True)
                        reward_df = r_data.iloc[i:self.t+1, :]
                        reward_df.iloc[0, 3] = reward_df.iloc[0,
                                                              3] + handling_fee
                        reward_df.iloc[-1, 3] = reward_df.iloc[-1,
                                                               3] - handling_fee - tax
                        reward_df["rate"] = reward_df["close"].pct_change()
                        mean_reward = reward_df.iloc[1:, :]['rate'].mean()
                        # 計算無風險利率
                        risk_free_rate = (
                            risk_free_rates / 365) * (self.t-i)
                        # 計算標準差
                        std_list = reward_df["close"]
                        std = np.std(std_list)
                        # 計算夏普值，並存進list
                        sharpe_ratio = ((mean_reward-risk_free_rate) / std)
                        sharpe_ratio_list.append(sharpe_ratio)

                    # reward為平均夏普值
                    sharpe = sum(sharpe_ratio_list) / len(sharpe_ratio_list)
                    if sharpe > 0:
                        reward = sharpe * 10
                    else:
                        reward = sharpe

                    record_reward = profits / sum(self.hold_value)
                    # 賣出將持有價格、部位、持有天數、持有總金額清空
                    self.positions.clear()
                    self.buy_size.clear()
                    self.hold_day.clear()
                    self.hold_value.clear()
                    self.positions = []
                    self.profits += profits

                    self.profits += profits
                    # 繪圖用
                    plot_data.iloc[self.t, 11] = 1

        # 有持有ETF就要扣管理費用
        if (len(self.hold_value) != 0):
            for i in self.hold_value:
                self.account_money -= i * (manage_fee / 365)

        # set next time
        self.t += 1
        self.position_value = 0
        self.position_value = self.position_value + \
            sum(self.hold_value) + self.account_money
        self.state = []
        self.history.pop(0)  # 移除歷史，可能是為了減少內存
        self.state.append(self.position_value)  # 加入新的時間
        self.state.append(self.data.iloc[self.t, :]["close"])  # 加入新的時間
        self.state.append(self.data.iloc[self.t, :]["MACD"])
        self.state.append(self.data.iloc[self.t, :]["SIGNAL"])
        self.state.append(self.data.iloc[self.t, :]["10 period VAMA"])
        self.state.append(self.data.iloc[self.t, :]["20 period VAMA"])
        self.history.append(self.state)

        return self.history, sharpe, record_reward
