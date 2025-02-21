from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
import wandb.sdk.data_types.video as wv
import pathlib
import numpy as np
import torch
import time
import json


class RealPushTLowDimRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir, n_obs_steps, n_action_steps, scene_config_path, init_humanstate_path, relative_actions, absolute_pos):
        super().__init__(output_dir)
        self.n_episodes_per_lumi = 1 # nb of episode
        self.luminosities = np.arange(1)+1 #np.arange(6)+1
        # self.luminosities = np.arange(6)+1 #np.arange(6)+1
        self.max_ts_per_episode = 500 #timesteps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.scene_config_path = scene_config_path
        self.init_humanstate_path = init_humanstate_path
        self.relative_actions = relative_actions
        self.absolute_pos = absolute_pos
    
    def run(self, policy: BaseLowdimPolicy):
        for lumi in self.luminosities:
            print("lumi : ", lumi)
            for i in range(self.n_episodes_per_lumi):
                print("episode : ", i)
                #time the episode
                t_before = time.time()
                self.run_episode(policy, lumi, i)
                t_after = time.time()
                print("episode duration (s) : ", t_after - t_before)
        return dict()

    def run_episode(self, policy: BaseLowdimPolicy, lumi, i):
        device = policy.device
        print("device : ", device)
        dtype = policy.dtype
        id_trial = wv.util.generate_id()
        path_trial = self.output_dir  + "/" + "lumi_{}/".format(lumi) + id_trial + "/"
        if not pathlib.Path(path_trial).exists():
            pathlib.Path(path_trial).mkdir(parents=True, exist_ok=True)
        laby_selected = 'A'
        actions_sequence = []
        actions_sequence_reconstructed = []
        obs_sequence = []
        scene = self.initialize_scene(scene_selected = laby_selected)
        human_state = self.initialize_human_state(scene_selected = laby_selected)
        t = 0
        lumis = np.array([[[lumi]]]).repeat(self.n_obs_steps, axis=1)
        prepared_obs = np.concatenate([scene, human_state, lumis], axis=2)
        # print("prepared_obs shape : ", prepared_obs.shape)
        # print("prepared_obs : ", prepared_obs)
        # time.sleep(1000)
        obs_sequence.append(prepared_obs)
        prepared_obs = torch.tensor(prepared_obs, device=device, dtype=dtype)
        while t < self.max_ts_per_episode:
            # print("t : ", t)
            #print t and flush 
            print("t : ", t, end="\r", flush=True)
            obs_dict = dict(obs=prepared_obs)
            time_before = time.time()
            policy_output = policy.predict_action(obs_dict)
            time_after = time.time()
            print("time for policy_output : ", time_after - time_before)
            time_before = time.time()
            action = policy_output['action_pred'].detach().cpu().numpy() #(1, 200, 9)
            # print("action delta : ", action[0,:5,:])
            # print('observation:', human_state)

            if self.relative_actions and self.absolute_pos:
                action_absolute = action[:,:,:9] # the first 9 absolute actions
                action_relative = action[:,:,9:] # the relative actions
                ### for absolute actions, we just take the predicted absolute actions
                actions_sequence.append(action_absolute)
                ### for relative actions, we reconstruct the absolute actions
                action_absolute_reconstructed = np.concatenate([human_state[:, -1:, :], action_relative[:, :, :]], axis=1)
                for idx in range(1, action_absolute_reconstructed.shape[1]):
                    action_absolute_reconstructed[:, idx, :] = action_absolute_reconstructed[:, idx, :] + action_absolute_reconstructed[:, idx-1, :]
                action_absolute_reconstructed = action_absolute_reconstructed[:, 1:, :]
                actions_sequence_reconstructed.append(action_absolute_reconstructed)

            # print("absolute position : ", action[0,:5,:])
            # time.sleep(100)
            # print(policy_output['action_pred'])
            # print(policy_output['action'])
            # actions_sequence.append(action)
            # we predict several steps ahead, so we need to update the human_state with all the actions :
            if self.n_action_steps < self.n_obs_steps: 
                human_state = np.concatenate([human_state[:, self.n_action_steps:, :], action], axis=1)
            else:
                # print(action.shape) # (1, 8, 9)
                # print(action[:, -self.n_obs_steps:, :].shape) 
                human_state = action_absolute_reconstructed[:, -self.n_obs_steps:, :] # (1, 8, 9)
            prepared_obs = np.concatenate([scene, human_state, lumis], axis=2)
            obs_sequence.append(prepared_obs)
            prepared_obs = torch.tensor(prepared_obs, device=device, dtype=dtype)
            t += self.n_action_steps
            time_after = time.time()
            print("time for prepare obs : ", time_after - time_before)
        #save obs_sequence and actions_sequence
        np.save(path_trial+'obs_sequence_lumi_{}_episode_{}.npy'.format(lumi, i), np.array(obs_sequence))
        np.save(path_trial+'actions_sequence_lumi_{}_episode_{}.npy'.format(lumi, i), np.array(actions_sequence))
        np.save(path_trial+'actions_sequence_reconstructed_lumi_{}_episode_{}.npy'.format(lumi, i), np.array(actions_sequence_reconstructed))

    def initialize_scene(self, scene_selected):
        ##### add labyrinth config here
        # load the json file
        with open(self.scene_config_path) as f:
            scene_config = json.load(f)
        lowdim_value = np.array(scene_config[scene_selected])
        # return an array (1, self.n_obs_steps, 9*3)
        lowdim_value = np.array([lowdim_value]*self.n_obs_steps).reshape(1, self.n_obs_steps, 9*3) #(1, self.n_obs_steps, 9*3)
        # return np.zeros((1, self.n_obs_steps, 9*3))
        return lowdim_value
    
    def initialize_human_state(self, scene_selected):
        ### human state :maybe random
        # with open(self.init_humanstate_path) as f:
        #     init_humanstate = json.load(f)
        # human_state = init_humanstate[scene_selected]
        # state_value = np.array(human_state['mean'])
        # state_value = np.array([state_value]*self.n_obs_steps).reshape(1, self.n_obs_steps, 9)
        # an example: [0.05169339   1.03437355  -0.12540438 0.15842498   0.79557971   0.27868093  47.15499046 -11.88175446 -6.37264612   2.]
        ex_array = np.array([0.05169339, 1.03437355, -0.12540438, 0.15842498, 0.79557971, 0.27868093, 47.15499046, -11.88175446, -6.37264612])
        state_value = np.array([ex_array]*self.n_obs_steps).reshape(1, self.n_obs_steps, 9)
        return state_value

