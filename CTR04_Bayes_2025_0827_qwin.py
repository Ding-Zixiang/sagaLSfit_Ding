#!/usr/bin/env python
# coding: utf-8

# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import itertools
import math
from matplotlib.colors import LogNorm
import time
#from numpy.random import *
import numexpr as ne
from IPython.display import clear_output
import os
import warnings
warnings.filterwarnings('ignore')
#from google.colab import drive
#drive.mount('/content/drive')


# In[17]:


import numpy as np

# ===== CTR Model #04 (closed-form) =====
# I(q) = 10**I0 * | sum_{n=0}^{NS} exp(-mu*n) * exp(i*q*aS*n)  +  1/(sqrt(2) * sin(q*a/2)) |^2
E_KEV_FIXED = 8.0    # fixed
A_FIXED     = 5.43   # Å, fixed
NS_FIXED    = 128971 # fixed

def _geom_series_complex(r, N):
    eps = 1e-12
    near1 = np.abs(r-1) < eps
    out = np.empty_like(r, dtype=np.complex128)
    out[near1]  = (N+1) + 0j
    out[~near1] = (1 - np.power(r[~near1], N+1)) / (1 - r[~near1])
    return out

def CTRmodel04(q_data, I0, mu, aS, a=A_FIXED, NS=NS_FIXED):
    r = np.exp(-mu) * np.exp(1j * q_data * aS)
    exp_sum = _geom_series_complex(r, NS)
    exp_sum = exp_sum + 1.0 / (np.sqrt(2.0) * np.sin(q_data * a / 2.0))
    Iq_ = (10.0**I0) * np.abs(exp_sum)**2
    return Iq_

def CTR04fit(q, _p_):
    # _p_: (num_temp, 3) -> [I0, mu, aS]
    I0 = _p_[:,0].reshape(num_temp,1)
    mu = _p_[:,1].reshape(num_temp,1)
    aS = _p_[:,2].reshape(num_temp,1)
    return CTRmodel04(q, I0, mu, aS)


# In[18]:


def axbfit(x,_p_):
    _a_ = _p_[:,0].reshape(num_temp,1)
    _b_ = _p_[:,1].reshape(num_temp,1)
    y = axb(x,_a_,_b_)
    return y


# In[19]:


# メトロポリス法によるパラメータ更新
def update_param(i,E,n,num_para,sigma_R,x,y,_p_,ratio_p,rslt_p,_rslt_p0_,_rslt_p1_,stepsize_p0,stepsize_p1,
                 _ratio_p_,num_temp,_p_min_,_p_max_,burn_in_length):
    #num_para = np.size(_p_max_)
    # temp=0はprior（事前分布）に基づいてサンプリング
    _p_[0,0] = (_p_max_[0]-_p_min_[0])*np.random.rand()+_p_min_[0]
    _p_[0,1] = (_p_max_[1]-_p_min_[1])*np.random.rand()+_p_min_[1]

    for _ii_ in range(num_para):
        _select_ = np.random.randint(0,num_para,num_temp) #パラメータ間の相関がサンプリングに影響しないようにランダムにパラメータ選択して更新。
        #_select_ = int(_ii_)*np.ones(num_temp,dtype="int8") # 順番にパラメータ更新する場合
        _next_p_ = 1*_p_
        for _j_ in range(1,num_temp):# temp=0は別のサンプリングを行う
            _ratio_p_[num_para*_j_+_select_[_j_]] = np.append(_ratio_p_[num_para*_j_+_select_[_j_]],0)
            if(_select_[_j_]==0):
                _next_p_[_j_,0] += stepsize_p0[_j_]*(2*np.random.rand()-1)
            if(_select_[_j_]==1):
                _next_p_[_j_,1] += stepsize_p1[_j_]*(2*np.random.rand()-1)

        _y_ = axbfit(x,_next_p_)
        next_E = 1/2/n/sigma_R**2*np.linalg.norm(y-(
                 _y_
                 ),axis=1)**2

        prob = np.exp(-n*temp*(next_E-E)) - np.random.rand(num_temp)
        J = np.where(prob>0)
        for _jj_ in J[0]:
            if(_p_min_[_select_[_jj_]]<= _next_p_[_jj_,_select_[_jj_]]<=_p_max_[_select_[_jj_]]):
                _p_[_jj_,_select_[_jj_]] = _next_p_[_jj_,_select_[_jj_]]
                E[_jj_] = next_E[_jj_]
                if(_jj_>0):
                    (_ratio_p_[num_para*_jj_+_select_[_jj_]])[-1] +=1#採択率についての履歴を残す
                if(i>=burn_in_length):
                    ratio_p[_jj_,_select_[_jj_]] += 1#採択率そのもの
        rslt_p[_select_[num_temp-1]] = np.append(rslt_p[_select_[num_temp-1]],_p_[num_temp-1,_select_[num_temp-1]])#最低温の履歴
        for _kk_ in range(1,num_temp):#全レプリカのサンプリング履歴を残す
            if(_select_[_kk_]==0):
                _rslt_p0_[_kk_] = np.append(_rslt_p0_[_kk_],_p_[_kk_,0])
            if(_select_[_kk_]==1):
                _rslt_p1_[_kk_] = np.append(_rslt_p1_[_kk_],_p_[_kk_,1])

    #temp=0は必ず更新
    E[0] = next_E[0]
    ratio_p[0,:] = 0
    _rslt_p0_[0] = np.append(_rslt_p0_[0],_p_[0,0])
    _rslt_p1_[0] = np.append(_rslt_p1_[0],_p_[0,1])

    return E


# In[20]:


# 擬似フォークト関数
# ローレンツ関数とガウス関数の幅を別々に設定
def psV(x,mu,w_l,w_g,A,x0):
    y = A*(mu*2/np.pi*w_l/(4*(x-x0)**2+w_l**2)+(1-mu)*np.sqrt(4*np.log(2))/np.sqrt(np.pi)/w_g*np.exp(-4*np.log(2)/w_g**2*(x-x0)**2))
    return y
# ローレンツ関数とガウス関数の幅を共通
#def psV(x,mu,w,A,x0):
#    y = A*(mu*2/np.pi*w/(4*(x-x0)**2+w**2)+(1-mu)*np.sqrt(4*np.log(2))/np.sqrt(np.pi)/w*np.exp(-4*np.log(2)/w**2*(x-x0)**2))
#    return y


# In[ ]:


def psVfit(x,_p_):
    A1   = _p_[:,0].reshape(num_temp,1)
    mu1  = _p_[:,1].reshape(num_temp,1)
    w_l1 = _p_[:,2].reshape(num_temp,1)
    w_g1 = _p_[:,3].reshape(num_temp,1)
    x1   = _p_[:,4].reshape(num_temp,1)
    A2   = _p_[:,5].reshape(num_temp,1)
    mu2  = _p_[:,6].reshape(num_temp,1)
    w_l2 = _p_[:,7].reshape(num_temp,1)
    w_g2 = _p_[:,8].reshape(num_temp,1)
    x2   = _p_[:,9].reshape(num_temp,1)
    y = psV(x,mu1,w_l1,w_g1,A1,x1) + psV(x,mu2,w_l2,w_g2,A2,x2)
    return y


# In[21]:


# ===== 读取实验数据 + 2θ→q，可视化窗口 =====
import pandas as pd, numpy as np, math, os, re

# 可视化 q 窗口（None 表示不限制；否则 (qmin, qmax)）
Q_WINDOW = (4.58, 4.69)

data_path = "data-2025-0822by akai/Si_TEOS_2thome.dat"

def load_si_teos(path):
    with open(path, "rb") as f:
        raw = f.read()
    txt = raw.decode("cp932", errors="ignore")
    rows = []
    for ln in txt.splitlines():
        if re.match(r"^\s*[-+]?\d+(\.\d+)?\s*,", ln):
            parts = [p.strip() for p in ln.split(",")]
            try:
                vals = [float(p.replace("E+", "e+").replace("E-", "e-")) for p in parts]
                rows.append(vals)
            except:
                pass
    df = pd.DataFrame(rows)
    if df.shape[1] >= 3:
        df = df.iloc[:, [0,2]].rename(columns={0:"two_theta_deg", 2:"intensity"})
    else:
        df = df.iloc[:, :2]
        df.columns = ["two_theta_deg", "intensity"]
    return df

df = load_si_teos(data_path)

lambda_ang = 12.39842 / 8.0
theta_rad  = np.deg2rad(df["two_theta_deg"].values / 2.0)
q = (4.0*np.pi/lambda_ang) * np.sin(theta_rad)
I_obs = df["intensity"].values.astype(np.float64)

mask  = np.isfinite(I_obs) & np.isfinite(q)
mask &= (np.abs(np.sin(q * A_FIXED / 2.0)) > 1e-6)

q_all = q[mask]; I_all = I_obs[mask]
if Q_WINDOW is not None:
    qmin, qmax = Q_WINDOW
    view_mask = (q_all>=qmin) & (q_all<=qmax)
else:
    view_mask = np.ones_like(q_all, dtype=bool)

x = q_all.reshape(-1)
y = I_all.reshape(-1)

# 输出目录（如果后面还会被 Cell 15 覆盖同名，也没关系）
_dir_ = globals().get("_dir_", "CTR04fit_2025_0827")
os.makedirs(_dir_, exist_ok=True)

plt.figure(figsize=(10,4))
plt.plot(q_all[view_mask], I_all[view_mask], ".", label="Observed (exp)")
plt.xlabel("q (1/Å)"); plt.ylabel("Intensity (a.u.)")
plt.legend()
if Q_WINDOW is not None: plt.xlim(Q_WINDOW)
plt.savefig(f"{_dir_}/01_exp_q.png", dpi=220, bbox_inches="tight")
plt.show()

print(f"[DATA] q-range(all) = [{x.min():.3f}, {x.max():.3f}] 1/Å, N={x.size}")
if Q_WINDOW is not None: print(f"[VIEW] q-window = {Q_WINDOW[0]:.3f}..{Q_WINDOW[1]:.3f} 1/Å")
print(f"Saved: {_dir_}/01_exp_q.png")


# In[23]:


x


# In[24]:


x


# In[ ]:





# In[25]:


y


# In[26]:


y


# In[ ]:





# In[27]:


# 解析するデータのプロット
plt.plot(x,y,"o")
plt.show()


# In[ ]:





# In[28]:


# -------- Parallel tempering MCMC (Metropolis) for CTRmodel04 --------
import time, os
np.random.seed(1)

# Parameters: 3 (I0, mu, aS)
num_para = 3

# Prior ranges (Uniform) -- adjust as needed
_p_min_ = np.array([  2.0, 1e-7, 5.00])   # I0, mu, aS
_p_max_ = np.array([  6.0, 1e-3, 6.00])

# Noise std estimate from data (robust): MAD
y_med = np.median(y)
_sigma_R_ = max(1e-6, 0.5 * np.median(np.abs(y - y_med)))
sigma_R   = _sigma_R_

# Parallel tempering schedule
num_temp = 50
temp = np.logspace(-4, 0, num_temp)  # β from 1e-4 to 1
temp[0] = 0  # prior-only

# Step sizes per parameter
_st = np.array([0.05, 1e-5, 5e-3])   # base widths (I0, mu, aS)
steps = np.zeros((num_temp, num_para))
for j in range(num_para):
    alpha = 5.0; d = 1.0
    steps[:, j] = np.where(temp <= 1/alpha, _st[j], _st[j]/(alpha*temp)**d)
steps[0, :] = _st

# Initial from prior
ini_pri = 1
if ini_pri == 1:
    _p_ = _p_min_ + (_p_max_ - _p_min_) * np.random.rand(num_temp, num_para)
else:
    _p_ = np.tile((_p_min_ + _p_max_)/2, (num_temp,1))

def neg_log_like(p, beta, q=x, y=y, sigma=sigma_R):
    yhat = CTRmodel04(q, p[0], p[1], p[2])
    resid = (y - yhat)
    return beta * (0.5/(sigma**2)) * np.sum(resid*resid)

def neg_log_prior(p):
    if np.any(p < _p_min_) or np.any(p > _p_max_):
        return np.inf
    return 0.0

def energy(p, beta):
    return neg_log_prior(p) + neg_log_like(p, beta)

cycle = 50000
burn_in_length = 10000
n = x.size

_dir_ = "CTR04fit_2025_0827"
os.makedirs(_dir_, exist_ok=True)

ratio_p = np.zeros((num_temp, num_para))
ratio_temp = np.zeros(num_temp)
mean_E = np.zeros(num_temp)

rslt_p = [np.array([]) for _ in range(num_para)]
_rslt_p0_ = [np.array([]) for _ in range(num_temp)]
_rslt_p1_ = [np.array([]) for _ in range(num_temp)]
_rslt_p2_ = [np.array([]) for _ in range(num_temp)]

_E_ = np.zeros(num_temp)
for j in range(num_temp):
    _E_[j] = energy(_p_[j], temp[j])

st = time.time()
thin = 20

for i in range(cycle):
    for _param_idx in np.random.permutation(num_para):
        for j in range(1, num_temp):
            cur = _p_[j].copy()
            prop = cur.copy()
            step = steps[j, _param_idx]
            prop[_param_idx] = cur[_param_idx] + np.random.normal(scale=step)
            low, high = _p_min_[_param_idx], _p_max_[_param_idx]
            if prop[_param_idx] < low or prop[_param_idx] > high:
                prop[_param_idx] = np.clip(low + (low - prop[_param_idx]) if prop[_param_idx] < low else high - (prop[_param_idx]-high), low, high)
            e_cur = _E_[j]
            e_prop = energy(prop, temp[j])
            if np.isfinite(e_prop) and (np.random.rand() < np.exp(-(e_prop - e_cur))):
                _p_[j] = prop
                _E_[j] = e_prop
                ratio_p[j, _param_idx] += 1
        _p_[0, _param_idx] = _p_min_[_param_idx] + (_p_max_[_param_idx]-_p_min_[_param_idx])*np.random.rand()
        _E_[0] = energy(_p_[0], temp[0])

    if (i % 2) == 0:
        pairs = [(j, j+1) for j in range(1, num_temp-1, 2)]
    else:
        pairs = [(j, j+1) for j in range(2, num_temp-1, 2)]
    for j, k in pairs:
        e_jj = _E_[j]; e_kk = _E_[k]
        e_jk = energy(_p_[j], temp[k]); e_kj = energy(_p_[k], temp[j])
        if np.log(np.random.rand()) < -(e_jk + e_kj - e_jj - e_kk):
            _p_[j], _p_[k] = _p_[k].copy(), _p_[j].copy()
            _E_[j], _E_[k] = _E_[k], _E_[j]
            ratio_temp[j] += 1; ratio_temp[k] += 1

    mean_E += _E_

    if (i % thin) == 0:
        for j in range(num_temp):
            _rslt_p0_[j] = np.append(_rslt_p0_[j], _p_[j,0])
            _rslt_p1_[j] = np.append(_rslt_p1_[j], _p_[j,1])
            _rslt_p2_[j] = np.append(_rslt_p2_[j], _p_[j,2])
        for pidx in range(num_para):
            rslt_p[pidx] = np.append(rslt_p[pidx], _p_[-1, pidx])

    if ((i+1) % 5000) == 0:
        print(i+1, "iters; elapsed", time.time()-st)
        np.save(_dir_+"/ratio_p.npy", ratio_p)
        np.save(_dir_+"/ratio_temp.npy", ratio_temp)
        np.save(_dir_+"/mean_E.npy", mean_E)
        np.save(_dir_+"/_rslt_p0_.npy", _rslt_p0_)
        np.save(_dir_+"/_rslt_p1_.npy", _rslt_p1_)
        np.save(_dir_+"/_rslt_p2_.npy", _rslt_p2_)
        np.save(_dir_+"/_E_.npy", _E_)
        np.save(_dir_+"/_p_all_.npy", _p_)
        np.save(_dir_+"/_mean_E_.npy", mean_E)
        np.save(_dir_+"/_ratio_p_.npy", ratio_p)

np.save(_dir_+"/ratio_p.npy", ratio_p)
np.save(_dir_+"/ratio_temp.npy", ratio_temp)
np.save(_dir_+"/mean_E.npy", mean_E)
np.save(_dir_+"/_rslt_p0_.npy", _rslt_p0_)
np.save(_dir_+"/_rslt_p1_.npy", _rslt_p1_)
np.save(_dir_+"/_rslt_p2_.npy", _rslt_p2_)
np.save(_dir_+"/_E_.npy", _E_)
np.save(_dir_+"/_p_all_.npy", _p_)
np.save(_dir_+"/_mean_E_.npy", mean_E)
np.save(_dir_+"/_ratio_p_.npy", ratio_p)
print("Total elapsed:", time.time()-st)


# In[29]:


_dir_


# In[ ]:





# In[30]:


# 採択率のプロット
ratio_p = np.load(_dir_+"/ratio_p.npy")
print("おおむね0.2から0.8の間に収まっていれば成功。")
for i in range(num_para):
    plt.plot(temp,ratio_p[:,i]/(cycle-burn_in_length),label="p{}".format(i))
    plt.xscale("log")
    plt.ylim(0,1)
    plt.xlabel("$b$")
    plt.ylabel("Acceptance ratio")
    plt.legend()
    plt.savefig(f"{_dir_}/02_acceptance_ratio_p{i}.png", dpi=220, bbox_inches="tight")
    plt.show()


# In[31]:


# 各パラメータの採択履歴をプロット
print("収束するまでの焼きなまし区間(burn_in)が十分か確認する。")
i = 44 # 履歴を確認したいレプリカのインデックス（逆温度の数が50の時は0から49までの番号）を入れる

_rslt_p0_ = np.load(_dir_+"/_rslt_p0_.npy", allow_pickle=True)
_rslt_p1_ = np.load(_dir_+"/_rslt_p1_.npy", allow_pickle=True)
_rslt_p2_ = np.load(_dir_+"/_rslt_p2_.npy", allow_pickle=True)

plt.figure(figsize=(10,3))
plt.plot(_rslt_p0_[i])
plt.title("Param0 (I0) trace, replica i={}".format(i))
plt.savefig(f"{_dir_}/03_trace_p0_rep{i}.png", dpi=220, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10,3))
plt.plot(_rslt_p1_[i])
plt.title("Param1 (mu) trace, replica i={}".format(i))
plt.savefig(f"{_dir_}/03_trace_p1_rep{i}.png", dpi=220, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10,3))
plt.plot(_rslt_p2_[i])
plt.title("Param2 (aS) trace, replica i={}".format(i))
plt.savefig(f"{_dir_}/03_trace_p2_rep{i}.png", dpi=220, bbox_inches="tight")
plt.show()


# In[32]:


# レプリカ交換率とmean_Eのプロット
ratio_temp = np.load(_dir_+"/ratio_temp.npy")
mean_E = np.load(_dir_+"/mean_E.npy")
print("最高温度のレプリカ(b=0)との交換率(右の値がゼロで無いことを確認)",ratio_temp[0]/((cycle-burn_in_length)/2))
_ratio_temp_ = np.zeros((num_temp,2))
_ratio_temp_[:,0] = 1*temp
_ratio_temp_[:,1] = 1*ratio_temp/((cycle-burn_in_length)/2)
_mean_E_ = np.zeros((num_temp,2))
_mean_E_[:,0] = 1*temp
_mean_E_[:,1] = 1*mean_E/(cycle-burn_in_length)

plt.plot(temp[1:num_temp-1],ratio_temp[1:num_temp-1]/((cycle-burn_in_length)/2))
plt.xscale("log")
plt.xlabel("$b$")
plt.ylabel("Exchange ratio")
plt.ylim(0,1)
plt.savefig(f"{_dir_}/04_exchange_ratio.png", dpi=220, bbox_inches="tight")
plt.show()
plt.plot(temp[1:num_temp],n*mean_E[1:num_temp]/(cycle-burn_in_length))
plt.xlabel("$b$")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("$N$ x $mean$_$E$")
plt.savefig(f"{_dir_}/04b_meanE.png", dpi=220, bbox_inches="tight")
plt.show()


# In[33]:


# ノイズ推定
F = []
for i in range(1,num_temp):
    A = 0
    for j in range(i):
        A += (temp[j+1] - temp[j])*mean_E[j+1]/(cycle-burn_in_length)*n
    F = np.append(F,A- n/2*(np.log(temp[i])-np.log(2*np.pi)))
plt.plot(temp[1:num_temp],F)
plt.xscale("log")
plt.vlines(temp[np.argmin(F)+1],np.min(F),np.max(F),linestyle="dotted")
#plt.yscale("log")
#plt.ylim(-200,500)
print("自由エネルギーの最小値",np.min(F))
print("推定されたノイズの標準偏差",1/(temp[np.argmin(F)+1])**0.5)
print("最適な逆温度のインデックス",np.argmin(F)+1)
plt.show()


# In[34]:


# 事後確率分布をプロット（最低温のレプリカを使用）
i = num_temp-1
_pp_MPM_ = np.zeros(num_para)
for j in range(num_para):
    arr = np.load(_dir_+"/_rslt_p{}_.npy".format(j), allow_pickle=True)
    series = (arr[i])[burn_in_length::]
    plt.title("p{} : std = {:.4f}".format(j, np.std(series)))
    plt.hist(series, bins=40)
    _pp_hist_ = np.histogram(series, bins=40)
    # bin center at maximum
    _pp_MPM_[j] = ((_pp_hist_[1])[np.argmax(_pp_hist_[0])+1] + (_pp_hist_[1])[np.argmax(_pp_hist_[0])]) / 2.0
    plt.vlines(_pp_MPM_[j], 0, 1.1*np.max(_pp_hist_[0]), linestyle="dotted", label="{:.4f}".format(_pp_MPM_[j]))
    plt.ylim(0, 1.1*np.max(_pp_hist_[0]))
    plt.legend()
    plt.savefig(f"{_dir_}/05_posterior_p{j}.png", dpi=220, bbox_inches="tight")
    plt.show()


# In[38]:


# パラメータごとの事後確率分布の最大値(MPM)によるフィッティング結果
_p_all_ = np.load(_dir_+"/_p_all_.npy")
_pp_MPM_ = np.median(_p_all_[-1:,:], axis=0)
I0, mu, aS = _pp_MPM_

y_fit = CTRmodel04(x, I0, mu, aS)

plt.figure(figsize=(8,4))
plt.plot(x, y, "o", label="Obs (exp)")
plt.plot(x, y_fit, "-", label="Fit (MPM)")
plt.xlabel("q (1/Å)"); plt.ylabel("Intensity (a.u.)")
if 'Q_WINDOW' in globals() and (Q_WINDOW is not None):
    plt.xlim(Q_WINDOW)
plt.legend()
plt.savefig(f"{_dir_}/06_fit_overlay.png", dpi=220, bbox_inches="tight")
plt.show()

resid = y - y_fit
plt.figure(figsize=(8,2.8))
plt.plot(x, resid, ".", label="Residual (Obs - Fit)")
plt.axhline(0, ls="--")
plt.xlabel("q (1/Å)"); plt.ylabel("Residual")
if 'Q_WINDOW' in globals() and (Q_WINDOW is not None):
    plt.xlim(Q_WINDOW)
plt.legend()
plt.savefig(f"{_dir_}/06b_residual.png", dpi=220, bbox_inches="tight")
plt.show()


# In[ ]:


_pp_MPM_


# In[ ]:


# MAP推定
_opt_j_ = np.argmin(F)+1
_pp_MAP_ = np.zeros(num_para)
_optimal_ = np.load(_dir_+"/_optimal_.npy")
for i in range(num_para):
    _pp_MAP_[i] = (_optimal_[num_para*_opt_j_+i])[cycle]
a   = _pp_MAP_[0]
b   = _pp_MAP_[1]
_y_ = axb(x,a,b)

plt.plot(x,y,"o",label="Data")
plt.plot(x,_y_,color="r",label="MAP")
plt.legend()


# In[ ]:


_pp_MAP_


# In[ ]:





# In[ ]:




