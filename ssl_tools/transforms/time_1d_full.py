from typing import Callable, List


import numpy as np

class Composer:
    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, dataset: np.ndarray, labels: np.ndarray = None):
        for t in self.transforms:
            dataset = t(dataset)
        return dataset, labels

    def __str__(self) -> str:
        return f"Composer of transforms: {[str(t) for t in self.transforms]}"

class ConcatComposer:
    def __init__(self, transforms: List[Callable], axis: int = 0):
        self.transforms = transforms
        self.axis = axis

    def __call__(self, dataset: np.ndarray, labels: np.ndarray = None):
        datas = [t(dataset) for t in self.transforms]
        if labels is not None:
            labels = np.resize(labels, len(datas))
        return np.concatenate(datas, axis=self.axis), labels

    def __str__(self) -> str:
        return f"ConcatComposer of transforms: {[str(t) for t in self.transforms]}"

class Identity:
    def __call__(self, dataset: np.ndarray):
        return dataset

class Scale:
    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def __call__(self, dataset: np.ndarray):
        factor = np.random.normal(loc=1., scale=self.sigma, size=(dataset.shape[0],dataset.shape[2]))
        data_scaled = np.multiply(dataset, factor[:,np.newaxis,:])
        return data_scaled

    def __str__(self) -> str:
        return f"Scale: sigma={self.sigma}"

class Rotate:
    def __call__(self, dataset: np.ndarray):
        flip = np.random.choice([-1, 1], size=(dataset.shape[0],dataset.shape[2]))
        rotate_axis = np.arange(dataset.shape[2])
        np.random.shuffle(rotate_axis)
        data_rotation = flip[:,np.newaxis,:] * dataset[:,:,rotate_axis]
        return data_rotation
    
    def __str__(self) -> str:
        return f"Rotate"

class Permutate:
    def __init__(self, max_segments: int = 5, segment_mode: str = "equal"):
        self.max_segments = max_segments
        self.segment_mode = segment_mode

    def __call__(self, dataset: np.ndarray):
        orig_steps = np.arange(dataset.shape[1])
        num_segs = np.random.randint(1, self.max_segments, size=(dataset.shape[0]))
        data_permute = np.zeros_like(dataset)
        for i, pat in enumerate(dataset):
            if num_segs[i] > 1:
                if self.segment_mode == "random":
                    split_points = np.random.choice(dataset.shape[1]-2, num_segs[i]-1, replace=False)
                    split_points.sort()
                    splits = np.split(orig_steps, split_points)
                else:
                    splits = np.array_split(orig_steps, num_segs[i])
                
                permutation_indexes = np.arange(len(splits))
                permutations = np.random.permutation(permutation_indexes)
                warp = np.concatenate([splits[permutation_indexes[p]] for p in permutations]).ravel()
                data_permute[i] = pat[warp]
            else:
                data_permute[i] = pat
        return data_permute

    def __str__(self) -> str:
        return f"Permutate: max_segments={self.max_segments}; segment_mode={self.segment_mode}"

class MagnitudeWarp:
    def __init__(self, sigma: float = 0.2, knot: int = 4):
        self.sigma = sigma
        self.knot = knot

    def __call__(self, dataset: np.ndarray):
        from scipy.interpolate import CubicSpline
        orig_steps = np.arange(dataset.shape[1])
        random_warps = np.random.normal(
            loc=1.0, scale=self.sigma, size=(dataset.shape[0], self.knot+2, dataset.shape[2])
        )
        warp_steps = (np.ones((dataset.shape[2],1))*(np.linspace(0, dataset.shape[1]-1., num=self.knot+2))).T
        data_m_Warp = np.zeros_like(dataset)
        for i, pat in enumerate(dataset):
            warper = np.array([CubicSpline(
                warp_steps[:,dim], random_warps[i,:,dim])(orig_steps)
                for dim in range(dataset.shape[2])]
            ).T
            data_m_Warp[i] = pat * warper
        return data_m_Warp

    def __str__(self) -> str:
        return f"MagnitudeWarp: sigma={self.sigma}; knot={self.knot}"

class TimeWarp:
    def __init__(self, sigma: float = 0.2, knot: int = 4):
        self.sigma = sigma
        self.knot = knot

    def __call__(self, dataset: np.ndarray):
        from scipy.interpolate import CubicSpline
        orig_steps = np.arange(dataset.shape[1])
        random_warps = np.random.normal(
            loc=1.0, scale=self.sigma, size=(dataset.shape[0], self.knot+2, dataset.shape[2]))
        warp_steps = (np.ones((dataset.shape[2],1))*(np.linspace(0, dataset.shape[1]-1., num=self.knot+2))).T
        data_t_Warp = np.zeros_like(dataset)
        for i, pat in enumerate(dataset):
            for dim in range(dataset.shape[2]):
                time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
                scale = (dataset.shape[1]-1)/time_warp[-1]
                data_t_Warp[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, dataset.shape[1]-1), pat[:,dim]).T
        return data_t_Warp

    def __str__(self) -> str:
        return f"TimeWarp: sigma={self.sigma}; knot={self.knot}"

class WindowSlice:
    def __init__(self, reduce_ratio: float = 0.9):
        self.reduce_ratio = reduce_ratio

    def __call__(self, dataset: np.ndarray):
        target_len = np.ceil(self.reduce_ratio*dataset.shape[1]).astype(int)
        if target_len >= dataset.shape[1]:
            return dataset
        starts = np.random.randint(low=0, high=dataset.shape[1]-target_len, size=(dataset.shape[0])).astype(int)
        ends = (target_len + starts).astype(int)
        data_w_Slice = np.zeros_like(dataset)
        for i, pat in enumerate(dataset):
            for dim in range(dataset.shape[2]):
                data_w_Slice[i,:,dim] = np.interp(np.linspace(0, target_len, num=dataset.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
        return data_w_Slice

    def __str__(self) -> str:
        return f"WindowSlice: reduce_ratio={self.reduce_ratio}"

class WindowWarp:
    def __init__(self, window_ratio=0.1, scales=[0.5, 2.]):
        self.window_ratio = window_ratio
        self.scales = scales

    def __call__(self, dataset: np.ndarray):
        warp_scales = np.random.choice(self.scales, dataset.shape[0])
        warp_size = np.ceil(self.window_ratio*dataset.shape[1]).astype(int)
        window_steps = np.arange(warp_size)
        window_starts = np.random.randint(low=1, high=dataset.shape[1]-warp_size-1, size=(dataset.shape[0])).astype(int)
        window_ends = (window_starts + warp_size).astype(int)
        data_w_Warp = np.zeros_like(dataset)
        for i, pat in enumerate(dataset):
            for dim in range(dataset.shape[2]):
                start_seg = pat[:window_starts[i],dim]
                window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
                end_seg = pat[window_ends[i]:,dim]
                warped = np.concatenate((start_seg, window_seg, end_seg))
                data_w_Warp[i,:,dim] = np.interp(np.arange(dataset.shape[1]), np.linspace(0, dataset.shape[1]-1., num=warped.size), warped).T
        return data_w_Warp

    def __str__(self) -> str:
        return f"WindowWarp: window_ratio={self.window_ratio}; scales={self.scales}"
