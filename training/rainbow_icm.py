import os
import time
import random
from collections import deque, namedtuple

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import TimeLimit

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image

# -----------------------------
# Hyperparameters
# -----------------------------
COPY_NETWORK_FREQ       = 10000
BUFFER_CAPACITY         = 10000
BATCH_SIZE              = 32
GAMMA                   = 0.9
EPS_START               = 1.0
EPS_END                 = 0.01
EPS_DECAY               = 0.9999
LEARNING_RATE           = 0.00025
ADAM_EPS                = 0.00015
PER_ALPHA               = 0.6
PER_BETA_START          = 0.4
PER_BETA_FRAMES         = 2000000
PER_EPSILON             = 0.1
N_STEP                  = 5
NOISY_SIGMA_INIT        = 2.5
# custom penalties
BACKWARD_PENALTY        = 0
STAY_PENALTY            = 0
DEATH_PENALTY           = -100
SKIP_FRAMES             = 4
MAX_EPISODE_STEPS       = 3000
MAX_FRAMES              = 44800000

# ICM hyperparams
ICM_EMBED_DIM           = 256
ICM_BETA                = 0.2   # inverse vs forward loss trade-off
ICM_ETA                 = 0.01  # intrinsic reward scale
ICM_LR                  = 1e-4

# -----------------------------
# 1. Environment Wrappers
# -----------------------------
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip
    def step(self, action):
        total_reward, done = 0.0, False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done: break
        return obs, total_reward, done, info

class GrayScaleResize(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.transform = T.Compose([
            T.ToPILImage(), T.Grayscale(), T.Resize((84,90)), T.ToTensor()
        ])
        self.observation_space = gym.spaces.Box(0.0,1.0,shape=(1,84,90),dtype=np.float32)
    def observation(self, obs):
        return self.transform(obs)

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k; self.frames = deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0,1,shape=(shp[0]*k,shp[1],shp[2]),dtype=np.float32)
    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k): self.frames.append(obs)
        return np.concatenate(self.frames,axis=0)
    def step(self,action):
        obs,r,done,info = self.env.step(action)
        self.frames.append(obs)
        return np.concatenate(self.frames,axis=0), r, done, info

def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, SKIP_FRAMES)
    env = GrayScaleResize(env)
    env = FrameStack(env, 4)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    return env

# -----------------------------
# 2. Intrinsic Curiosity Module
# -----------------------------
class ICM(nn.Module):
    def __init__(self, feat_dim, n_actions, embed_dim=ICM_EMBED_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(feat_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU()
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(embed_dim*2,512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.forward_model = nn.Sequential(
            nn.Linear(embed_dim + n_actions,512), nn.ReLU(),
            nn.Linear(512, embed_dim)
        )
    def forward(self, feat, next_feat, action):
        phi = self.encoder(feat)
        phi_next = self.encoder(next_feat)
        inv_in = torch.cat([phi,phi_next],dim=1)
        logits = self.inverse_model(inv_in)
        a_onehot = F.one_hot(action, logits.size(-1)).float()
        fwd_in = torch.cat([phi,a_onehot],dim=1)
        pred_phi_next = self.forward_model(fwd_in)
        return logits, pred_phi_next, phi_next

# -----------------------------
# 3. Noisy Linear Layer
# -----------------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_f, out_f, sigma_init=NOISY_SIGMA_INIT):
        super().__init__(); self.in_f, self.out_f = in_f, out_f
        self.weight_mu = nn.Parameter(torch.empty(out_f,in_f))
        self.weight_sigma = nn.Parameter(torch.empty(out_f,in_f))
        self.register_buffer('weight_epsilon',torch.empty(out_f,in_f))
        self.bias_mu = nn.Parameter(torch.empty(out_f))
        self.bias_sigma = nn.Parameter(torch.empty(out_f))
        self.register_buffer('bias_epsilon',torch.empty(out_f))
        self.sigma_init = sigma_init; self.reset_parameters(); self.reset_noise()
    def reset_parameters(self):
        bound=1/(self.in_f**0.5)
        nn.init.uniform_(self.weight_mu,-bound,bound)
        nn.init.constant_(self.weight_sigma,self.sigma_init/(self.in_f**0.5))
        nn.init.uniform_(self.bias_mu,-bound,bound)
        nn.init.constant_(self.bias_sigma,self.sigma_init/(self.out_f**0.5))
    def reset_noise(self):
        f=lambda x: x.sign()*x.abs().sqrt()
        eps_in=f(torch.randn(self.in_f)); eps_out=f(torch.randn(self.out_f))
        self.weight_epsilon.copy_(eps_out.ger(eps_in)); self.bias_epsilon.copy_(eps_out)
    def forward(self,x):
        if self.training:
            w=self.weight_mu+self.weight_sigma*self.weight_epsilon
            b=self.bias_mu+self.bias_sigma*self.bias_epsilon
        else: w,b=self.weight_mu,self.bias_mu
        return F.linear(x,w,b)

# -----------------------------
# 4. Dueling CNN
# -----------------------------
class DuelingCNN(nn.Module):
    def __init__(self,in_c,n_actions):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(in_c,32,8,4),nn.ReLU(),
            nn.Conv2d(32,64,4,2),nn.ReLU(),
            nn.Conv2d(64,64,3,1),nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy=torch.zeros(1,in_c,84,90)
            feat_dim=self.features(dummy).shape[1]
        self.val_noisy=NoisyLinear(feat_dim,512); self.val=NoisyLinear(512,1)
        self.adv_noisy=NoisyLinear(feat_dim,512); self.adv=NoisyLinear(512,n_actions)
    def forward(self,x):
        x=self.features(x/255.0)
        v=F.relu(self.val_noisy(x)); v=self.val(v)
        a=F.relu(self.adv_noisy(x)); a=self.adv(a)
        return v+(a-a.mean(dim=1,keepdim=True))
    def reset_noise(self):
        for m in [self.val_noisy,self.val,self.adv_noisy,self.adv]: m.reset_noise()

# -----------------------------
# 5. Replay Buffer
# -----------------------------
class PrioritizedReplayBuffer:
    def __init__(self,cap,alpha,beta_start,beta_frames,n_step,gamma):
        self.cap,self.alpha=cap,alpha; self.beta_start,self.beta_frames=beta_start,beta_frames
        self.beta_by_frame=lambda f:min(1.0,beta_start+f*(1.0-beta_start)/beta_frames)
        self.n_step,self.gamma=n_step,gamma; self.buffer=[]
        self.prios=np.zeros((cap,),dtype=np.float32); self.pos=0; self.n_buf=deque(maxlen=n_step)
        self.Exp=namedtuple('Exp',['s','a','r','s2','d'])
    def _get_n_step(self):
        r,s2,d=self.n_buf[-1].r,self.n_buf[-1].s2,self.n_buf[-1].d
        for trans in reversed(list(self.n_buf)[:-1]):
            r=trans.r+self.gamma*r*(1-trans.d); s2,d=(trans.s2,trans.d) if trans.d else (s2,d)
        return r,s2,d
    def add(self,s,a,r,s2,d):
        self.n_buf.append(self.Exp(s,a,r,s2,d));
        if len(self.n_buf)<self.n_step: return
        r_n,s2_n,d_n=self._get_n_step();s0,a0=self.n_buf[0].s,self.n_buf[0].a
        exp=self.Exp(s0,a0,r_n,s2_n,d_n)
        if len(self.buffer)<self.cap:
            self.buffer.append(exp); prio=1.0 if len(self.buffer)==1 else self.prios.max()
        else:
            self.buffer[self.pos]=exp; prio=self.prios.max()
        self.prios[self.pos]=prio; self.pos=(self.pos+1)%self.cap
    def sample(self,bs,frame_idx):
        N=len(self.buffer)
        if N==0: return [],[],[],[],[],[],[]
        prios=self.prios[:N]**self.alpha; sum_p=prios.sum()
        probs=prios/sum_p if sum_p>0 else np.ones_like(prios)/N
        idxs=np.random.choice(N,bs,p=probs)
        batch=self.Exp(*zip(*[self.buffer[i] for i in idxs]))
        beta=self.beta_by_frame(frame_idx)
        weights=(N*probs[idxs])**(-beta); weights/=weights.max()
        return (np.array(batch.s),batch.a,batch.r,np.array(batch.s2),batch.d,weights.astype(np.float32),idxs)
    def update_priorities(self,idxs,errors):
        for i,e in zip(idxs,errors): self.prios[i]=abs(e)+1e-6

# -----------------------------
# 6. Agent with ICM
# -----------------------------
class Agent:
    def __init__(self,obs_shape,n_actions,device):
        self.device=device; self.n_actions=n_actions
        self.online=DuelingCNN(obs_shape[0],n_actions).to(device)
        self.target=DuelingCNN(obs_shape[0],n_actions).to(device)
        self.target.load_state_dict(self.online.state_dict()); self.target.eval()
        self.opt=optim.Adam(self.online.parameters(),lr=LEARNING_RATE,eps=ADAM_EPS)
        # ICM
        with torch.no_grad():
            dummy=torch.zeros(1,*obs_shape).to(device)
            feat_dim=self.online.features(dummy).shape[1]
        self.icm=ICM(feat_dim,n_actions).to(device)
        self.icm_opt=optim.Adam(self.icm.parameters(),lr=ICM_LR)
        self.buffer=PrioritizedReplayBuffer(BUFFER_CAPACITY,PER_ALPHA,PER_BETA_START,PER_BETA_FRAMES,N_STEP,GAMMA)
        self.gamma=GAMMA; self.batch_size=BATCH_SIZE; self.frame_idx=0; self.update_freq=COPY_NETWORK_FREQ
    def act(self,state):
        s_t=torch.tensor(state,dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad(): q=self.online(s_t)
        return int(q.argmax(1).item())
    def push(self,s,a,r,s2,d): self.buffer.add(s,a,r,s2,d)
    def learn(self):
        if self.frame_idx<self.batch_size: return
        # sample
        s,a,r_ext,s2,d,w,idxs=self.buffer.sample(self.batch_size,self.frame_idx)
        s=torch.tensor(s,dtype=torch.float32,device=self.device)
        s2=torch.tensor(s2,dtype=torch.float32,device=self.device)
        a=torch.tensor(a,dtype=torch.int64,device=self.device)
        d=torch.tensor(d,dtype=torch.float32,device=self.device)
        w=torch.tensor(w,dtype=torch.float32,device=self.device)
        r_ext=torch.tensor(r_ext,dtype=torch.float32,device=self.device)
        # features
        feat=self.online.features(s/255.0)
        nxt_feat=self.online.features(s2/255.0)
        feat_icm=feat.detach()
        nxt_feat_icm=nxt_feat.detach()
        # ICM forward
        logits,pred_phi_n,true_phi_n=self.icm(feat_icm,nxt_feat_icm,a)
        inv_loss=F.cross_entropy(logits,a)
        fwd_loss=F.mse_loss(pred_phi_n,true_phi_n.detach())
        icm_loss=(1-ICM_BETA)*inv_loss+ICM_BETA*fwd_loss
        with torch.no_grad():
            int_r=ICM_ETA*0.5*(pred_phi_n-true_phi_n).pow(2).sum(dim=1)
        # DQN targets
        q_pred=self.online(s).gather(1,a.unsqueeze(1)).squeeze(1)
        a_n=self.online(s2).argmax(1)
        q_next=self.target(s2).gather(1,a_n.unsqueeze(1)).squeeze(1)
        total_r=r_ext+int_r
        q_tar=total_r + (self.gamma**N_STEP)*q_next*(1-d)
        td=q_pred-q_tar.detach()
        dqn_loss=(F.smooth_l1_loss(q_pred,q_tar.detach(),reduction='none')*w).mean()
        # update DQN
        self.opt.zero_grad(); dqn_loss.backward(); self.opt.step()
        self.online.reset_noise(); self.target.reset_noise()
        self.buffer.update_priorities(idxs,td.detach().cpu().numpy())
        # update ICM
        self.icm_opt.zero_grad(); icm_loss.backward(); self.icm_opt.step()
        # sync
        if self.frame_idx%self.update_freq==0: self.target.load_state_dict(self.online.state_dict())

# -----------------------------
# 7. Training Loop
# -----------------------------
def train(num_episodes, checkpoint_path='checkpoints/rainbow_icm/rainbow_icm.pth'):
    env=make_env()
    agent=Agent(env.observation_space.shape, env.action_space.n, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    start_ep,fi=1,0
    if os.path.isfile(checkpoint_path):
        ck=torch.load(checkpoint_path,map_location=agent.device)
        agent.online.load_state_dict(ck['model']);agent.target.load_state_dict(ck['model'])
        agent.opt.load_state_dict(ck['optimizer']);agent.icm_opt.load_state_dict(ck['icm_opt'])
        fi=ck.get('frame_idx',0);start_ep=ck.get('episode',0)+1
    agent.frame_idx=fi
    # warm-up
    raw=env.reset(); state=raw
    while len(agent.buffer.buffer)<BATCH_SIZE:
        a=agent.act(state)
        nxt, r, d, _=env.step(a)
        nxt_state=nxt
        agent.push(state,a,r,nxt_state,d)
        state=nxt_state
        if d: state=env.reset()
    # history
    hist={'reward':[], 'env_reward':[], 'stage':[], 'dur':[], 'batches':[], 'Trun':[]}
    chunk=time.time()
    for ep in range(start_ep,num_episodes+1):
        obs=env.reset(); state=obs; ep_r,ep_er,prev_x,prev_life=0,0,None,None;done=False
        while not done:
            agent.frame_idx+=1
            a=agent.act(state)
            nxt, r_env, done, info=env.step(a)
            truncated=info.get('TimeLimit.truncated',False)
            done_flag=done and not truncated
            cr=r_env
            x_pos,life=info.get('x_pos'),info.get('life')
            if x_pos is not None:
                if prev_x is None:prev_x=x_pos
                dx=x_pos-prev_x; cr+=BACKWARD_PENALTY if dx<0 else STAY_PENALTY if dx==0 else 0; prev_x=x_pos
            if prev_life is None: prev_life=life
            elif life<prev_life: cr+=DEATH_PENALTY; prev_life=life
            nxt_state=nxt
            agent.push(state,a,cr,nxt_state,done_flag)
            agent.learn()
            state=nxt_state; ep_r+=cr; ep_er+=r_env

        status = "TERMINATED" if done else "TRUNCATED"
        hist['env_reward'].append(ep_r); hist['env_reward'].append(ep_er); hist['stage'].append(env.unwrapped._stage)
        hist['Trun'].append(status)
        if ep%100==0:
            dur=time.time()-chunk;chunk=time.time()
            ae=np.mean(hist['env_reward'][-100:]);ar=np.mean(hist['reward'][-100:])
            avg_stage = np.mean(hist['stage'][-100:])
            truncated = hist['Trun'][-100:].count('TRUNCATED')

            print(f"[Batch {ep//100} Ep {ep}] | EnvR {ae:.2f} | CustR {ar:.2f} | Stage {avg_stage} | Trun.Num {truncated} | Time {dur/60:.2f}min")
            hist['batches'].append(ep)
            torch.save({'model':agent.online.state_dict(),'optimizer':agent.opt.state_dict(),
                        'icm_opt':agent.icm_opt.state_dict(),'frame_idx':agent.frame_idx,'episode':ep},checkpoint_path)

            chunks = len(hist['env_reward']) // 100
            xs     = [i*100 for i in range(1, chunks+1)]
            avg_envs  = [np.mean(hist['env_reward'][(i-1)*100:i*100]) for i in range(1, chunks+1)]
            avg_custs = [np.mean(hist['env_reward'][(i-1)*100:i*100]) for i in range(1, chunks+1)]

            plt.figure(figsize=(8,4))
            plt.plot(xs, avg_envs,  marker='o', label='Env Reward')
            plt.plot(xs, avg_custs, marker='x', label='Custom Reward')
            plt.xlabel('Episodes')
            plt.ylabel('Avg Reward per 100 eps')
            plt.title('Reward Comparison')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(
                os.path.dirname(checkpoint_path),
                'reward_comparison.png'))
            plt.close()

            # 5) 新增：10 次 inference 评估
            eval_env = make_env()
            eval_rewards = []
            for _ in range(10):
                e_obs = eval_env.reset()
                done = False
                total = 0.0
                step = 0
                while not done and step < 2000:
                    a = agent.act(e_obs)
                    e_obs, r, done, _ = eval_env.step(a)
                    total += r
                    step += 1
                eval_rewards.append(total)
            eval_env.close()
            print(f"    → Eval avg over 10 eps: {np.mean(eval_rewards):.2f}")

    print("Training complete.")
    return hist

if __name__=='__main__':
    train(num_episodes=100000)
