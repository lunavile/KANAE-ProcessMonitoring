import time
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from tqdm import tqdm
from scipy import stats

from src.utils import set_seed, prepare_data_from_train_points
from src.config import BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY


def benchmark_training(
    model_fn,
    df,
    n_simulations,
    seed,
    reps: int = 5,
    warmup_batches: int = 10,
    device_mode: str = "cuda",
):
    """
    Benchmark forward/backward time and memory for a model on CPU/GPU (no AMP).

    Args:
        model_fn: callable -> model constructor (lambda input_size: model_fn(input_size, **cfg))
        df: training dataframe
        n_simulations: number of samples to use
        seed: random seed
        reps: number of *timed* batches (after warm-up)
        warmup_batches: number of initial batches to skip from timing
        device_mode: 'cpu', 'cuda', or 'both'

    Returns:
        dict: {'cpu': {...}, 'gpu': {...}} depending on mode
    """
    set_seed(seed)
    results = {}

    # === Prepare dataset ===
    X_train_scaled, _, _ = prepare_data_from_train_points(df, n_simulations, seed, sim_len=500)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    input_size = X_train_scaled.shape[1]

    # === Internal benchmark function ===
    def _run_benchmark(device: torch.device):
        model = model_fn(input_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        forward_times, backward_times, forward_mems, backward_mems = [], [], [], []
        model.train()

        print(
            f"🧪 Benchmarking on {device} | warm-up: {warmup_batches} | measured batches: {reps}"
        )

        for i, batch in enumerate(tqdm(loader, desc=f"[{device}]")):
            # stop when enough timed batches have been collected
            if reps is not None and i >= warmup_batches + reps:
                break
            
            inputs = batch[0].to(device)
            optimizer.zero_grad(set_to_none=True)

            # === Warm-up phase (skip timing) ===
            if i < warmup_batches:
                _ = model(inputs)
                loss = criterion(_[0] if isinstance(_, tuple) else _, inputs)
                loss.backward()
                optimizer.step()
                continue

            # === Forward timing ===
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                start_f, end_f = torch.cuda.Event(True), torch.cuda.Event(True)
                start_f.record()
                out = model(inputs)
                end_f.record()
                torch.cuda.synchronize()
                fwd_time = start_f.elapsed_time(end_f)
                fwd_mem = torch.cuda.max_memory_allocated() / (1024**3)
            else:
                t0 = time.time()
                out = model(inputs)
                fwd_time = (time.time() - t0) * 1000
                fwd_mem = 0.0

            loss = criterion(out[0] if isinstance(out, tuple) else out, inputs)

            # === Backward timing ===
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                start_b, end_b = torch.cuda.Event(True), torch.cuda.Event(True)
                start_b.record()
                loss.backward()
                end_b.record()
                torch.cuda.synchronize()
                bwd_time = start_b.elapsed_time(end_b)
                bwd_mem = torch.cuda.max_memory_allocated() / (1024**3)
            else:
                t1 = time.time()
                loss.backward()
                bwd_time = (time.time() - t1) * 1000
                bwd_mem = 0.0

            optimizer.step()

            forward_times.append(fwd_time)
            backward_times.append(bwd_time)
            forward_mems.append(fwd_mem)
            backward_mems.append(bwd_mem)

        # === Compute mean and 95% confidence interval ===
        def mean_ci(x):
            x = np.array(x, dtype=np.float64)
            n = len(x)
            if n < 2:
                return float(np.mean(x)), 0.0
            mean, std = np.mean(x), np.std(x, ddof=1)
            ci = stats.t.ppf(0.975, n - 1) * std / np.sqrt(n)
            return float(mean), float(ci)

        fwd_mean, fwd_ci = mean_ci(forward_times)
        bwd_mean, bwd_ci = mean_ci(backward_times)
        fwd_mem_mean, fwd_mem_ci = mean_ci(forward_mems)
        bwd_mem_mean, bwd_mem_ci = mean_ci(backward_mems)

        return {
            "forward_ms": (fwd_mean, fwd_ci),
            "backward_ms": (bwd_mean, bwd_ci),
            "forward_mem_GB": (fwd_mem_mean, fwd_mem_ci),
            "backward_mem_GB": (bwd_mem_mean, bwd_mem_ci),
            "params": int(sum(p.numel() for p in model.parameters())),
            "trainable_params": int(
                sum(p.numel() for p in model.parameters() if p.requires_grad)
            ),
            "warmup_batches": warmup_batches,
            "timed_batches": len(forward_times),
        }

    # === Run for requested devices ===
    if device_mode in ("cpu", "both"):
        results["cpu"] = _run_benchmark(torch.device("cpu"))
    if device_mode in ("cuda", "gpu", "both"):
        if torch.cuda.is_available():
            results["gpu"] = _run_benchmark(torch.device("cuda"))
        else:
            print("⚠️ CUDA not available, skipping GPU benchmark.")

    return results



