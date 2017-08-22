import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
import random
import copy

"""
まず，強化学習を使って問題を解くには，解きたい問題（”環境”と呼びます）をしっかり定義する必要があります．環境の定義の仕方は，OpenAIが公開している強化学習ベンチマーク環境のGym（https://github.com/openai/gym）のインタフェースに従っています．Gymの環境で動かすこともできますし，インタフェースを揃えればオリジナルな環境で動かすこともできます．基本的にはresetとstepという2つのメソッドが実装されていれば十分です．
"""

# ぶっちゃけこっちを読んだほうが良い
# http://qiita.com/uezo/items/87b25c93199d72a56a9a

# インプットデータ。52 = 11*2*2+4（各プレイヤーの所持金0-10、スコア0-10、目の前にある数字1,2,3,4 and 1,2,3,4の既出）
glob_inpXdim = 52

# ouput 11通り
glob_outdim = 11

# Qfunctionの定義
# http://qiita.com/masataka46/items/7729a74d8dc15de7b5a8

#explorer用のランダム関数オブジェクト
class RandomActor:
    def __init__(self):
        pass
    def random_action_func(self):
        # 所持金を最大値にしたランダムを返すだけ
        return random.randint(0,10)

ra = RandomActor()
    
class CustomDiscreteQFunction(chainer.Chain):
    # ネットワークの形を定める
    def __init__(self):
        super().__init__(l1=L.Linear(glob_inpXdim, 50),
                         l2=L.Linear(50, 11))
    def __call__(self, x, test=False):
        h = F.relu(self.l1(x))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))

# seriを行うagent
class seriAgent:
    def __init__(self):
        pass
    def allzero(self):
        return 0
    def loadAgent(self,fname):
        # optimizer, qfuncをagentに入れ込む
        self.q_func = CustomDiscreteQFunction()
        self.optimizer = chainer.optimizers.Adam(eps=1e-2)
        self.optimizer.setup(self.q_func)

        # この辺、公式チュートリアルのコピペ。問題によって調整...知らんがな
        # https://github.com/chainer/chainerrl/blob/master/examples/quickstart/quickstart.ipynb
        self.gamma = 0.95
        self.explorer = chainerrl.explorers.ConstantEpsilonGreedy(
                epsilon=0.05, random_action_func = ra.random_action_func)
        self.replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
        self.phi = lambda x: x.astype(np.float32, copy=False)
        
        self.agent = chainerrl.agents.DoubleDQN(
                self.q_func, self.optimizer, self.replay_buffer, self.gamma, self.explorer,
                replay_start_size=500,  phi=self.phi)
        
        if fname != "":
            print("load agent from : " + fname)
            self.agent.load(fname)
    def domove(self,param):
        pass

# seriに関する環境
class seriEnv:
    def __init__(self):
        self.random2p = False
        self.reset()
        self.win = 0
        self.lose = 0
        self.draw = 0
    def reset(self):
        # reset は環境をリセットして現在の観測を返す
        self.pos = np.zeros(glob_inpXdim)
        for i in range(glob_inpXdim):
            self.pos[i] = 0
        self.p1 = 0 # 点数を0にする
        self.p2 = 0 # 点数を0にする
        self.m1 = 10 # 所持金を10にする
        self.m2 = 10 # 所持金を10にする
        self.ply = 0 # 手番を0にする
        # 初手のカードはreset時点で決定させておく
        self.app = self.selectcard(self.pos,self.ply)
        self.pos[44+self.app] = 1 # ボードのカードとして表示する
        self.pos[48+self.app] = 1 # 既出にする
        self.setmp2pos()
        return self.pos

    def step(self, move):
        if move > self.m1:
            move = self.m1
        if self.random2p == True:
            # 合法手をランダムで選ぶ
            movee = self.domove_random(1)
        else:
            # movee（敵の指し手）はランダムや過去の学習データから決める
            movee = self.enemy.agent.act(self.reversePos(self.pos))
            # 所持金以上の手を返した場合は所持金全てを競りに出させる
            if movee > self.m2:
                movee = self.m2
        
        # step は環境にアクションを送り，4つの値（次の観測，報酬，エピソード終端かどうか，追加情報）を返す
        # 打たれた手によって点数と所持金を更新
        if move > movee:
            self.m1 -= move
            self.p1 += self.app+1
        elif movee > move:
            self.m2 -= movee
            self.p2 += self.app+1
        #print("player 1 = "+ str(self.m1) + "(money), " + str(self.p1) + "(score)  " + "player 2 = "+ str(self.m2) + "(money), " + str(self.p2) + "(score)" )

        # 次のカードを出させる
        self.ply += 1

        reward = 0.0
        done = False
        if self.ply == 4:
            done =  True
            # ゲーム終了時に勝っていればrewardをあたえる
            if self.p1 > self.p2:
                reward = 1.0
                self.win += 1
            elif self.p1 < self.p2:
                reward = -1.0
                self.lose += 1
            elif self.p1 == self.p2:
                self.draw += 1
        else:
            # ゲームが終了してないなら次のカードを出す
            self.app = self.selectcard(self.pos,self.ply)
            for j in range(4):
                self.pos[44+j] = 0
            self.pos[44+self.app] = 1 # ボードのカードとして表示する
            self.pos[48+self.app] = 1 # 既出にする
            self.setmp2pos()
        info = "hello"
        return self.pos, reward, done, info
    
    def setmp2pos(self):
        # 点数と所持金を反映させる
        # どの勝利点が既出かの更新はplayout内部で行うことに注意
        for i in range(44):
            self.pos[i] = 0
        self.pos[self.m1] = 1
        self.pos[11+self.p1] = 1
        self.pos[22+self.m2] = 1
        self.pos[33+self.p2] = 1

    def selectcard(self, pos, ply):
        # まだ出してない勝利点の中から一つをランダムで選択
        app = random.randint(1,4-ply)
        for i in range(4):
            if pos[48+i] == 0:
                app -= 1
                if app == 0:
                    return i
        print("no valid card")
        print(pos)
        return -1

    def reversePos(self, posin):
        # 2p用の盤面を生成（変数を反転させる）
        posout = copy.deepcopy(posin)
        for i in range(22):
            posout[i] = posin[22+i]
            posout[i+22] = posin[i]
        return posout
            
    def domove_random(self,pid):
        if pid == 0:
            param = self.pos
        else:
            param = self.reversePos(self.pos)
        # サンドバックとしてランダムムーブを用意しておく
        for i in range(glob_outdim):
            if param[i] == 1:
                return random.randint(0,i)
        print("invalid randommove")
        return -1

    def loadEnemy(self, fname):
        if fname == "random":
            self.random2p = True
            return
        # プレイヤーの初期化
        self.enemy = seriAgent()
        self.enemy.loadAgent(fname)

# 学習 or 対戦
Learning = False

# NNからなるエージェント（こいつはどの道必要）
sagent = seriAgent()
# 学習させる場合の初期値（になってるか怪しい
sagent.loadAgent("model2")

env = seriEnv()
# 学習させる場合の対戦相手
env.loadEnemy("random")

# Training
obs = env.reset()
r = 0
done = False
for loop in range(10000):
    #print("newgame dazoi")
    while not done:
        if Learning == True:
            action = sagent.agent.act_and_train(obs, r)
        else:
            #action = sagent.agent.act_and_train(obs, r)
            action = sagent.agent.act(obs)
        obs, r, done, info = env.step(action)
            
    if Learning == True:
        sagent.agent.stop_episode_and_train(obs, r, done)
            
    obs = env.reset()
    r = 0
    done = False
    if loop % 100 == 0:
        print(str(env.win)+"-"+str(env.draw)+"-"+str(env.lose))
if Learning == True:
    # この名前で保存する
    sagent.agent.save('model2')
print("done")
