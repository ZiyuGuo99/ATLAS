import os

import numpy as np
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _is_qwen3vl_model(pretrain: str) -> bool:
    """Check if the model is a Qwen3-VL model by inspecting config."""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(pretrain, trust_remote_code=True)
        # Qwen3-VL models typically have model_type "qwen3_vl" or similar
        model_type = getattr(config, 'model_type', '').lower()
        if 'qwen3' in model_type and 'vl' in model_type:
            return True
        # Also check if it's a Qwen3-VL by checking the config structure
        # Qwen3-VL might not have text_config attribute
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'num_attention_heads'):
            return False  # Has text_config, likely Qwen2/2.5-VL
        # If model_type contains qwen and doesn't have standard text_config, might be Qwen3-VL
        if 'qwen' in model_type:
            return 'qwen3' in model_type or not hasattr(config, 'text_config')
    except Exception as e:
        logger.warning(f"Could not check model type for {pretrain}: {e}")
    return False


@ray.remote
def get_all_env_variables():
    import os

    return os.environ


@ray.remote
class LLMRayActor:

    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        if kwargs.get("distributed_executor_backend") == "ray":
            # a hack to make the script work.
            # stop ray from manipulating CUDA_VISIBLE_DEVICES
            # at the top-level when the distributed_executor_backend is ray.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # every worker will use 0.2 GPU, so that we can schedule
        # 2 instances on the same GPUs.
        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = "0.2"
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            print(f"creating LLM with bundle_indices={bundle_indices}")

        # Number of actors that will send prompt to this engine
        self.num_actors = kwargs.pop("num_actors")
        self.actor_counter = 0
        self.requests = {}
        self.responses = {}

        # Check if this is a Qwen3-VL model before trying to load with vLLM
        model_path = kwargs.get("model", args[0] if args else None)
        if model_path and _is_qwen3vl_model(model_path):
            logger.warning("=" * 80)
            logger.warning("WARNING: Detected Qwen3-VL model, but vLLM may not support it yet!")
            logger.warning(f"Model path: {model_path}")
            logger.warning("If you encounter AssertionError about 'text_config.num_attention_heads',")
            logger.warning("this means your vLLM version doesn't support Qwen3-VL's config structure.")
            logger.warning("=" * 80)
            logger.warning("Possible solutions:")
            logger.warning("1. Update vLLM to a version that supports Qwen3-VL")
            logger.warning("2. Use a Qwen2.5-VL or Qwen2-VL model instead")
            logger.warning("3. Use a different inference backend (not vLLM)")
            logger.warning("=" * 80)

        try:
            self.llm = LLM(*args, **kwargs)
        except (AssertionError, ValueError, Exception) as e:
            error_str = str(e)
            error_type = type(e).__name__
            
            # Handle "No model architectures are specified" error
            if "No model architectures are specified" in error_str or "model architectures" in error_str.lower():
                import sys
                vllm_version = "unknown"
                try:
                    import vllm
                    vllm_version = getattr(vllm, '__version__', 'unknown')
                except:
                    pass
                
                error_msg = f"""
{'=' * 80}
CRITICAL ERROR: vLLM cannot detect model architecture.

Model path: {model_path}
vLLM version: {vllm_version}
Error type: {error_type}
Error: {error_str}

This error occurs because:
- Newer vLLM versions require explicit model architecture detection
- The model config might not have 'architectures' field properly set
- vLLM v1 engine is stricter about architecture validation

SOLUTIONS:
1. Check the model's config.json has 'architectures' field:
   python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('{model_path}', trust_remote_code=True); print('Architectures:', c.architectures)"

2. Try setting VLLM_USE_V1=0 to use legacy engine (already set in eval.sh)

3. Use a model with properly configured architecture field

4. Downgrade vLLM to a version that auto-detects architecture better

For now, evaluation cannot proceed with this model using current vLLM version.
{'=' * 80}
"""
                print(error_msg, file=sys.stderr, flush=True)
                logger.error(error_msg)
                raise RuntimeError(
                    f"vLLM cannot detect model architecture for: {model_path}. "
                    f"See stderr for detailed error message."
                ) from e
            
            # Handle original text_config error
            if "text_config" in error_str or "num_attention_heads" in error_str:
                # Print directly to stderr so it's visible in Ray logs
                import sys
                vllm_version = "unknown"
                try:
                    import vllm
                    vllm_version = getattr(vllm, '__version__', 'unknown')
                except:
                    pass
                
                error_msg = f"""
{'=' * 80}
CRITICAL ERROR: vLLM cannot load Qwen3-VL model due to config structure mismatch.

Model path: {model_path}
vLLM version: {vllm_version}
Original error: {error_str}

This error occurs because:
- Qwen3-VL uses a different config structure than Qwen2/2.5-VL
- vLLM expects 'config.text_config.num_attention_heads' which Qwen3-VL doesn't have
- Your vLLM version doesn't support Qwen3-VL yet

SOLUTIONS:
1. Use a Qwen2.5-VL or Qwen2-VL model instead (compatible with current vLLM)
   
2. Wait for vLLM to add Qwen3-VL support (check vLLM release notes)

3. Use transformers directly without vLLM (not implemented in this framework)

For now, evaluation cannot proceed with this Qwen3-VL model using vLLM.
{'=' * 80}
"""
                print(error_msg, file=sys.stderr, flush=True)
                logger.error(error_msg)
                raise RuntimeError(
                    f"vLLM does not support Qwen3-VL model: {model_path}. "
                    f"Please use a Qwen2.5-VL or Qwen2-VL model instead. "
                    f"See stderr for detailed error message."
                ) from e
            raise

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray):
        return self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.llm.collective_rpc("update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache))

    def reset_prefix_cache(self):
        self.llm.llm_engine.reset_prefix_cache()

    def sleep(self, level=1):
        self.llm.sleep(level=level)

    def wake_up(self):
        self.llm.wake_up()

    def add_requests(self, actor_rank, *, sampling_params, prompt_token_ids):
        """
        Save the requests from actors and generate responses when all actors have sent their requests
        """
        self.requests[actor_rank] = prompt_token_ids
        self.actor_counter += 1
        if self.actor_counter == self.num_actors:
            assert len(self.requests) == self.num_actors
            num_requests = []
            requests = []
            for actor_rank, request in self.requests.items():
                num_requests.append((actor_rank, len(request)))
                requests.extend(request)

            if len(requests) > 0:
                # For now we assume that all requests have the same sampling params
                responses = self.llm.generate(sampling_params=sampling_params, prompt_token_ids=requests)
            else:
                responses = []

            offset = 0
            self.responses = {}
            for actor_rank, num in num_requests:
                self.responses[actor_rank] = responses[offset : offset + num]
                offset += num

            self.actor_counter = 0
            self.requests = {}

    def add_requests_vlm(self, actor_rank, *, sampling_params, vllm_vision_input):
        """
        Save the requests from actors and generate responses when all actors have sent their requests
        """
        self.requests[actor_rank] = vllm_vision_input
        self.actor_counter += 1
        if self.actor_counter == self.num_actors:
            assert len(self.requests) == self.num_actors, f"{len(self.requests)} != {self.num_actors}"
            num_requests = []
            requests = []
            for actor_rank, request in self.requests.items():
                num_requests.append((actor_rank, len(request)))
                requests.extend(request)

            if len(requests) > 0:
                # For now we assume that all requests have the same sampling params
                responses = self.llm.generate(requests, sampling_params=sampling_params)
            else:
                responses = []

            offset = 0
            self.responses = {}
            for actor_rank, num in num_requests:
                self.responses[actor_rank] = responses[offset : offset + num]
                offset += num

            self.actor_counter = 0
            self.requests = {}
            
    def add_requests_vlm_mix(self, actor_rank, *, sampling_params, vllm_vision_input):
        """
        Save the requests from actors and generate responses when all actors have sent their requests
        """
        self.requests[actor_rank] = vllm_vision_input
        self.actor_counter += 1
        if self.actor_counter == self.num_actors:
            assert len(self.requests) == self.num_actors, f"{len(self.requests)} != {self.num_actors}"
            num_requests = []
            requests = []
            vrall, trall = [], []
            vrsrc, trsrc = [], []
            self.responses = {}
            for actor_rank, request in self.requests.items():
                vreq, treq = request
                if vreq: 
                    vrall.extend(vreq)
                    vrsrc.extend([actor_rank] * len(vreq))
                    # vresponses = self.llm.generate(vreq, sampling_params=sampling_params)
                    # print('!!!! debug vr', type(vresponses))
                # else:
                #     vresponses = []
                if treq: 
                    trall.extend(treq)
                    trsrc.extend([actor_rank] * len(treq))
                    # tresponses = self.llm.generate(treq, sampling_params=sampling_params)
                    # print('!!!! debug tr', type(tresponses))
                   
            vresponses = self.llm.generate(vrall, sampling_params=sampling_params)
            tresponses = self.llm.generate(sampling_params=sampling_params, prompt_token_ids=trall)
            for actor_rank, request in self.requests.items():
                self.responses[actor_rank] = []
            for rank, rsp in zip(vrsrc, vresponses):
                self.responses[rank].append(rsp)
            for rank, rsp in zip(trsrc, tresponses):
                self.responses[rank].append(rsp)
            print('debug inside vllm engine')
                
            self.actor_counter = 0
            self.requests = {}

    def get_responses(self, actor_rank):
        """
        Return the responses for the actor with the given rank
        """
        return self.responses.pop(actor_rank)


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    num_total_actors: int,
    shared_pg=None,
    gpu_memory_utilization=None,
    vllm_enable_sleep=False,
):
    import vllm

    # assert vllm.__version__ >= "0.7.0", "OpenRLHF only supports vllm >= 0.7.0"

    vllm_engines = []
    num_gpus = int(tensor_parallel_size == 1)
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    print(f"===> [verbose] creating {num_engines} vllm engines")
    for i in range(num_engines):
        bundle_indices = None
        scheduling_strategy = None

        # Hybrid engine
        if shared_pg is not None:
            assert vllm.__version__ >= "0.7.2", "Only vllm >= 0.7.2 supports hybrid engine"

            if tensor_parallel_size > 1:
                scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=shared_pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=i * tensor_parallel_size
                )
                bundle_indices = np.arange(i * tensor_parallel_size, (i + 1) * tensor_parallel_size).tolist()
            else:
                num_gpus = 0.2
                scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=shared_pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=i
                )
        # Distributed RLHF
        elif tensor_parallel_size > 1:
            bundles = [{"GPU": 1, "CPU": 1}] * tensor_parallel_size
            pg = placement_group(bundles)
            ray.get(pg.ready())

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0
            )

        if num_engines >= num_total_actors:
            num_actors = 1
        else:
            num_actors = num_total_actors // num_engines + int(i < num_total_actors % num_engines)
        print(f"====> warning {os.getenv('MAX_PIXELS')}")
        
        # Check if this is a Qwen3-VL model and add appropriate parameters
        is_qwen3vl = _is_qwen3vl_model(pretrain)
        
        # Try to get model architecture from config (for newer vLLM that requires it)
        model_arch = None
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(pretrain, trust_remote_code=True)
            architectures = getattr(config, 'architectures', None)
            if architectures and len(architectures) > 0:
                model_arch = architectures[0]
                logger.info(f"Detected model architecture: {model_arch} for model: {pretrain}")
        except Exception as e:
            logger.warning(f"Could not detect model architecture for {pretrain}: {e}")
        
        vllm_kwargs = {
            "model": pretrain,
            "enforce_eager": enforce_eager,
            "worker_cls": "openrlhf.trainer.ray.vllm_worker_wrap.WorkerWrap",
            "tensor_parallel_size": tensor_parallel_size,
            "seed": seed,  # Fixed seed for all engines (was: seed + i)
            "distributed_executor_backend": distributed_executor_backend,
            "max_model_len": max_model_len,
            "enable_prefix_caching": enable_prefix_caching,
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "num_actors": num_actors,
            "gpu_memory_utilization": gpu_memory_utilization,
            "bundle_indices": bundle_indices if shared_pg else None,
            "enable_sleep_mode": vllm_enable_sleep,
            "limit_mm_per_prompt": {"image": 24+1},
            "mm_processor_kwargs": {
                "min_pixels": int(os.getenv("MIN_PIXELS", 256 * 28 * 28)),
                "max_pixels": int(os.getenv("MAX_PIXELS", 5120 * 28 * 28)),
            },
        }
        
        # For newer vLLM versions, try to disable v1 engine if VLLM_USE_V1=0
        vllm_use_v1 = os.getenv("VLLM_USE_V1", "0")
        if vllm_use_v1 == "0":
            logger.info(f"VLLM_USE_V1=0, attempting to use legacy engine (vLLM version: {vllm.__version__})")
            # Some newer vLLM versions might need explicit parameter to disable v1
            # Check if the parameter exists before adding it
            try:
                import inspect
                llm_signature = inspect.signature(vllm.LLM.__init__)
                if 'use_v1' in llm_signature.parameters:
                    vllm_kwargs["use_v1"] = False
                    logger.info("Added use_v1=False to vLLM kwargs")
            except:
                pass
        
        if is_qwen3vl:
            logger.warning(f"Detected Qwen3-VL model: {pretrain}")
            logger.warning("Note: Qwen3-VL may require specific vLLM version or parameters.")
            logger.warning("If you encounter AssertionError about text_config, your vLLM version may not support Qwen3-VL yet.")
            # Qwen3-VL might need different parameters - try without text_config assumptions
            # Some vLLM versions might need explicit model type or other parameters
        
        vllm_engines.append(
            LLMRayActor.options(
                num_cpus=0,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(**vllm_kwargs)
        )

    return vllm_engines
