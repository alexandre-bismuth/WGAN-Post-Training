from pathlib import Path
from typing import Callable, Optional, TypeVar, Dict, Tuple, List, Union
from s5.dataloading import make_data_loader
from .lobster_dataloader import LOBSTER, LOBSTER_Dataset, LOBSTER_Sampler
# from lob.encoding import Message_Tokenizer



DEFAULT_CACHE_DIR_ROOT = Path('./cache_dir/')
DATA_DIR = Path('../data/')

DataLoader = TypeVar('DataLoader')
InputType = [str, Optional[int], Optional[int]]
ReturnType = Tuple[LOBSTER, DataLoader, DataLoader, DataLoader, Dict, int, int, int, int, int, int]

# Custom loading functions must therefore have the template.
dataset_fn = Callable[[str, Optional[int], Optional[int]], ReturnType]


def create_lobster_prediction_dataset(
		cache_dir: Union[str, Path] = DATA_DIR,
		seed: int = 42,
		mask_fn = LOBSTER_Dataset.no_mask,
		msg_seq_len: int = 500,
		micro_bsz: int=128,
		num_devices: int=4,
		tp_size: int=1,
		use_book_data: bool = False,
		use_simple_book: bool = False,
		book_transform: bool = False,
		book_ablation: str = 'real',
		book_depth: int = 500,
		n_data_workers: int = 0,
		return_raw_msgs: bool = False,
		shuffle_train=True,
		rand_offset=True,
		debug_overfit=False,
		val_split: float = 0.01,
		test_split: float = 0.1,
		pin_memory: bool = True,
		prefetch_factor: int = 2,
		persistent_workers: bool = True,
		test_dir_name: Union[str, Path, None] = None,
		use_distributed_sampler: bool = False,
		process_rank: int = 0,
		process_count: int = 1,
		# Multi-ticker support
		tickers: Optional[List[str]] = None,
		data_root: Optional[str] = None,
		train_date_range: Optional[tuple] = None,
		test_date_range: Optional[tuple] = None,
		token_mode: str = '24tok',
	) -> ReturnType:
	""" 
	"""
	if debug_overfit:
		rand_offset= False
		shuffle_train= False


	print("[*] Generating LOBSTER Prediction Dataset from", cache_dir)
	from .lobster_dataloader import LOBSTER
	name = 'lobster'

	dataset_obj = LOBSTER(
		name,
		data_dir=cache_dir,
		mask_fn=mask_fn,
		msg_seq_len=msg_seq_len,
		use_book_data=use_book_data,
		use_simple_book=use_simple_book,
		book_transform=book_transform,
		book_ablation=book_ablation,
		book_depth=book_depth,
		# Cache stores fully-decompressed in-memory numpy arrays (not mmap).
		# At ~360 MB/file (msg+book combined for mega-cap) × 250 files ~= 90 GB
		# per worker — multiplied by N_DATA_WORKERS ranks this is the OOM driver
		# observed at N≥4 (job 4405666 MaxRSS 466 GB/rank). With multi-ticker
		# global shuffle the cache hit rate is already ~0.2% even at 250, so
		# dropping to 8 (~3 GB/worker) is essentially free for throughput.
		n_cache_files=8,
		return_raw_msgs=return_raw_msgs,
		rand_offset=rand_offset,
		debug_overfit=debug_overfit,
		val_split=val_split,
		test_split=test_split,
		test_data_dir=test_dir_name,
		# Multi-ticker
		tickers=tickers,
		data_root=data_root,
		train_date_range=train_date_range,
		test_date_range=test_date_range,
		token_mode=token_mode,
	)
	dataset_obj.setup()
 
	# breakpoint()

	print("Using mask function:", mask_fn)

	# use sampler to only get individual samples and automatic batching from dataloader
	#trn_sampler = LOBSTER_Sampler(
	#		dataset_obj.dataset_train, n_files_shuffle=5, batch_size=1, seed=seed)
	
	# With TP, intra-node GPUs share the same batch (head-parallel, not data-parallel).
	# For cross-node TP (tp_size > num_devices), all GPUs in a node are in one TP group,
	# so each node loads exactly micro_bsz samples (dp_devices_per_node = 1).
	dp_devices = max(1, num_devices // tp_size) if tp_size > 1 else num_devices
	per_process_bsz = micro_bsz * dp_devices
	trn_loader = create_lobster_train_loader(
		dataset_obj, seed, per_process_bsz, n_data_workers, reset_train_offsets=rand_offset, shuffle=shuffle_train,
		pin_memory=pin_memory, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
		use_distributed_sampler=use_distributed_sampler, process_rank=process_rank, process_count=process_count)
	# NOTE: drop_last=True recompiles the model for a smaller batch size
	val_sampler = None
	tst_sampler = None
	if use_distributed_sampler and process_count > 1:
		from torch.utils.data import DistributedSampler
		val_sampler = DistributedSampler(
			dataset_obj.dataset_val, num_replicas=process_count,
			rank=process_rank, shuffle=False, drop_last=True)
		tst_sampler = DistributedSampler(
			dataset_obj.dataset_test, num_replicas=process_count,
			rank=process_rank, shuffle=False, drop_last=True)
		print(f"[*] Val DistributedSampler: rank={process_rank}/{process_count}, "
			  f"samples_per_node={len(val_sampler)}")
		print(f"[*] Test DistributedSampler: rank={process_rank}/{process_count}, "
			  f"samples_per_node={len(tst_sampler)}")
	val_loader = make_data_loader(
		dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=per_process_bsz,
		drop_last=True, shuffle=False, sampler=val_sampler, num_workers=n_data_workers,
		pin_memory=pin_memory, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
	tst_loader = make_data_loader(
		dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=per_process_bsz,
		drop_last=True, shuffle=False, sampler=tst_sampler, num_workers=n_data_workers,
		pin_memory=pin_memory, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.L
	IN_DIM = dataset_obj.d_input
	TRAIN_SIZE = len(dataset_obj.dataset_train)
	aux_loaders = {}

	# Per-ticker test loaders (multi-ticker mode)
	# IMPORTANT: 488 per-ticker test loaders × N_DATA_WORKERS workers each was
	# a fork bomb (~488 × 4 = 1952 worker procs/rank). Force num_workers=0 +
	# persistent_workers=False so these only run synchronously during eval.
	# The per-ticker test datasets themselves carry n_cache_files=0 (set in
	# LOBSTER.setup) so even synchronous reads are memory-cheap.
	if dataset_obj.per_ticker_test_datasets:
		per_ticker_test = {}
		for ticker, tk_dataset in dataset_obj.per_ticker_test_datasets.items():
			tk_sampler = None
			if use_distributed_sampler and process_count > 1:
				from torch.utils.data import DistributedSampler
				tk_sampler = DistributedSampler(
					tk_dataset, num_replicas=process_count,
					rank=process_rank, shuffle=False, drop_last=True)
			per_ticker_test[ticker] = make_data_loader(
				tk_dataset, dataset_obj, seed=seed, batch_size=per_process_bsz,
				drop_last=True, shuffle=False, sampler=tk_sampler, num_workers=0,
				pin_memory=pin_memory, prefetch_factor=None, persistent_workers=False)
		aux_loaders['per_ticker_test'] = per_ticker_test
		print(f"[*] Per-ticker test loaders: {len(per_ticker_test)} tickers (num_workers=0, persistent=False)")

	BOOK_SEQ_LEN = dataset_obj.L_book
	BOOK_DIM = dataset_obj.d_book

	return (dataset_obj, trn_loader, val_loader, tst_loader, aux_loaders,
	 		N_CLASSES, SEQ_LENGTH, IN_DIM, BOOK_SEQ_LEN, BOOK_DIM, TRAIN_SIZE)

def create_lobster_train_loader(dataset_obj, seed, per_process_bsz, num_workers, reset_train_offsets=False, shuffle=True,
								pin_memory=True, prefetch_factor=2, persistent_workers=True,
								use_distributed_sampler=False, process_rank=0, process_count=1,
								resume_from_step=None, resume_epoch=0):
	"""Create train DataLoader, optionally skipping already-completed batches at sampler level.

	When resume_from_step is set, the sampler indices are sliced to exclude
	the first `resume_from_step * per_process_bsz` samples.  This avoids
	the ~3h idle skip that previously used `continue` in the training loop.
	"""
	if reset_train_offsets:
		dataset_obj.reset_train_offsets()

	train_sampler = None
	if use_distributed_sampler and process_count > 1:
		from torch.utils.data import DistributedSampler
		train_sampler = DistributedSampler(
			dataset_obj.dataset_train,
			num_replicas=process_count,
			rank=process_rank,
			shuffle=shuffle,
			seed=seed,
			drop_last=True,
		)
		# Reproduce the same shuffle order as the original run
		train_sampler.set_epoch(resume_epoch)
		print(f"[*] DistributedSampler: rank={process_rank}/{process_count}, "
			  f"samples_per_node={len(train_sampler)}")
		shuffle = False  # sampler handles shuffling

		# Sampler-level skip: slice off completed batches so DataLoader
		# never calls __getitem__ for them (zero IO overhead).
		#
		# DESIGN NOTE: Current approach materializes all indices via list(train_sampler)
		# then slices [skip_samples:]. Memory cost = N_samples/N_nodes × 28 bytes:
		#   128N: ~12 MB (negligible), 2N: ~756 MB (acceptable for testing).
		# Both approaches require O(N) randperm — the bottleneck is shuffle order
		# reproduction, not the list materialization.
		#
		# Alternative (MaxText/Grain): Google's MaxText uses Grain library with
		# ArrayRecordDataSource (O(1) random access) + iterator state serialization
		# via get_state()/set_state(). The iterator checkpoint is a ~few KB JSON
		# file per process, and resume is O(1) index seek with no randperm.
		# See: AlphaTrade/maxtext/src/MaxText/checkpointing.py (GrainCheckpointHandler)
		# This requires migrating data format from .npy to ArrayRecord — not worth
		# the effort for current dataset sizes.
		#
		# PERF (benchmarked 2026-03-03, 54M samples):
		#   randperm(54M): 2.6s CPU | tolist(): 0.5s | slice: 0.1s | total: ~3.2s
		#   All CPU-only, zero GPU waste. Fully masked by JAX coordinator init (2-5 min).
		#   vs pre-fix continue-based skip: ~3h wall + 192 GPU-hours wasted (3375x slower).
		if resume_from_step is not None and resume_from_step > 0:
			skip_samples = resume_from_step * per_process_bsz
			full_indices = list(train_sampler)
			if skip_samples < len(full_indices):
				remaining_indices = full_indices[skip_samples:]
				from torch.utils.data.sampler import SequentialSampler
				# Use a simple list-based sampler that yields indices in order
				# (order is already determined by DistributedSampler + epoch seed)
				train_sampler = remaining_indices  # DataLoader accepts a list as sampler
				print(f"[Resume] Sampler skip: {skip_samples}/{len(full_indices)} samples skipped "
					  f"({resume_from_step} batches × {per_process_bsz} BSZ), "
					  f"{len(remaining_indices)} remaining")
			else:
				print(f"[Resume] WARNING: skip_samples={skip_samples} >= total={len(full_indices)}, "
					  f"no data remaining — epoch already complete")
				return None
	else:
		# Non-distributed: handle resume skip for sequential/random sampler
		if resume_from_step is not None and resume_from_step > 0:
			skip_samples = resume_from_step * per_process_bsz
			total_samples = len(dataset_obj.dataset_train)
			if shuffle:
				import torch
				g = torch.Generator()
				g.manual_seed(seed + resume_epoch)
				full_indices = torch.randperm(total_samples, generator=g).tolist()
			else:
				full_indices = list(range(total_samples))
			# drop_last equivalent: truncate to multiple of batch size
			full_indices = full_indices[:total_samples - total_samples % per_process_bsz]
			if skip_samples < len(full_indices):
				remaining_indices = full_indices[skip_samples:]
				train_sampler = remaining_indices
				shuffle = False  # indices already in correct order
				print(f"[Resume] Sampler skip (non-distributed): {skip_samples}/{len(full_indices)} samples skipped, "
					  f"{len(remaining_indices)} remaining")
			else:
				print(f"[Resume] WARNING: skip_samples={skip_samples} >= total={len(full_indices)}, "
					  f"no data remaining")
				return None

	trn_loader = make_data_loader(
		dataset_obj.dataset_train,
		dataset_obj,
		seed=seed,
		batch_size=per_process_bsz,
		shuffle=shuffle,
		drop_last=True,
		sampler=train_sampler,
		num_workers=num_workers,
		worker_init_fn=force_cpu,
		pin_memory=pin_memory,
		prefetch_factor=prefetch_factor,
		persistent_workers=persistent_workers)
	return trn_loader

Datasets = {
	# financial data
	"lobster-prediction": create_lobster_prediction_dataset,
}


def force_cpu(index:int):
	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
	os.environ["JAX_PLATFORMS"] = "cpu"