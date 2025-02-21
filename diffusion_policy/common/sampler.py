from typing import Optional
import numpy as np
import numba
from diffusion_policy.common.replay_buffer import ReplayBuffer
import json
import time


@numba.jit(nopython=True)
def create_indices(
    episode_ends:np.ndarray, sequence_length:int, 
    episode_mask: np.ndarray,
    pad_before: int=0, pad_after: int=0,
    debug:bool=True) -> np.ndarray:
    episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert(start_offset >= 0)
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask

def get_val_mask_split_lumi(n_episodes, val_ratio_traj, val_ratio_lumi, seed=0, fold=True, metadata_path=None, n_lumi=2, lumi_val=[2,4,6]):
    #load json
    val_mask_lumi = np.zeros(n_episodes, dtype=bool)
    val_mask_lumi_unseen_0 = np.zeros(n_episodes, dtype=bool)
    val_mask_lumi_unseen_1 = np.zeros(n_episodes, dtype=bool)
    val_mask_lumi_unseen_2 = np.zeros(n_episodes, dtype=bool)
    val_mask_traj = np.zeros(n_episodes, dtype=bool)
    #random choose val_ratio_lumi among range(0, n_lumi)
    rng = np.random.default_rng(seed=seed)
    val_lumi_idxs = lumi_val#rng.choice(n_lumi, size=round(n_lumi * val_ratio_lumi), replace=False)+1
    print("val_lumi_idxs : ", val_lumi_idxs)
    with open(metadata_path) as f: # read metadata for choosing lumi
        metadata = json.load(f)
        print("metadata : ", metadata)
        
        for i in range(n_episodes):
            l = metadata['E'+str(i)]['lumi']
            if l in val_lumi_idxs:
                val_mask_lumi[i] = True
                if l == val_lumi_idxs[0]:
                    val_mask_lumi_unseen_0[i] = True
                elif l == val_lumi_idxs[1]:
                    val_mask_lumi_unseen_1[i] = True
                elif l == val_lumi_idxs[2]:
                    val_mask_lumi_unseen_2[i] = True

        #find false value indexes in val_mask_lumi
        indxes_false = np.where(val_mask_lumi == False)
        #random choose val_ratio_traj among range(0, len(indxes_false))
        val_traj_idxs = rng.choice(len(indxes_false[0]), size=round(len(indxes_false[0]) * val_ratio_traj), replace=False)
        for i in val_traj_idxs:
            val_mask_traj[indxes_false[0][i]] = True
        
        ### nb of val_mask_lumi == the sum of val_mask_lumi_unseen_0, val_mask_lumi_unseen_1, val_mask_lumi_unseen_2
        ### choose val_ratio_lumi percentage of each unseen lumi for validation
        indexes_false_unseen_0 = np.where(val_mask_lumi_unseen_0 == True)
        indexes_false_unseen_1 = np.where(val_mask_lumi_unseen_1 == True)
        indexes_false_unseen_2 = np.where(val_mask_lumi_unseen_2 == True)
        val_lumi_idxs_unseen_0 = rng.choice(len(indexes_false_unseen_0[0]), size=round(len(indexes_false_unseen_0[0]) * (1-val_ratio_lumi)), replace=False)
        val_lumi_idxs_unseen_1 = rng.choice(len(indexes_false_unseen_1[0]), size=round(len(indexes_false_unseen_1[0]) * (1-val_ratio_lumi)), replace=False)
        val_lumi_idxs_unseen_2 = rng.choice(len(indexes_false_unseen_2[0]), size=round(len(indexes_false_unseen_2[0]) * (1-val_ratio_lumi)), replace=False)
        for i in val_lumi_idxs_unseen_0:
            val_mask_lumi_unseen_0[indexes_false_unseen_0[0][i]] = False
        for i in val_lumi_idxs_unseen_1:
            val_mask_lumi_unseen_1[indexes_false_unseen_1[0][i]] = False
        for i in val_lumi_idxs_unseen_2:
            val_mask_lumi_unseen_2[indexes_false_unseen_2[0][i]] = False
    reduced_val_mask_lumi = val_mask_lumi_unseen_0 | val_mask_lumi_unseen_1 | val_mask_lumi_unseen_2

    # print("val_mask_lumi : ", val_mask_lumi)
    # print("val_mask_traj : ", val_mask_traj)
    # print("reduced_val_mask_lumi : ", reduced_val_mask_lumi)

    # return val_mask_lumi | val_mask_traj, val_mask_lumi, val_mask_traj
    return val_mask_lumi | val_mask_traj, reduced_val_mask_lumi, val_mask_traj
        
def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask

class SequenceSampler:
    def __init__(self, 
        replay_buffer: ReplayBuffer, 
        sequence_length:int,
        pad_before:int=0,
        pad_after:int=0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray]=None,
        ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert(sequence_length >= 1)
        if keys is None:
            keys = list(replay_buffer.keys())
        
        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(episode_ends, 
                sequence_length=sequence_length, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
                )
        else:
            indices = np.zeros((0,4), dtype=np.int64)

        # print("indices : ", indices)
        # input()
        

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices 
        self.keys = list(keys) # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
    
    def __len__(self):
        return len(self.indices)
        
    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full((n_data,) + input_arr.shape[1:], 
                    fill_value=np.nan, dtype=input_arr.dtype)
                try:
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
                except Exception as e:
                    import pdb; pdb.set_trace()
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result
