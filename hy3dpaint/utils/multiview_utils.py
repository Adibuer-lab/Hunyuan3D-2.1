# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import json
import time
import torch
import random
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple
import huggingface_hub
from omegaconf import OmegaConf
from diffusers import DiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler, DDIMScheduler, UniPCMultistepScheduler


class multiviewDiffusionNet:
    def __init__(self, config) -> None:
        self.device = config.device
        allow_downloads = os.getenv("HUNYUAN_HF_ALLOW_DOWNLOADS", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
        local_files_only = not allow_downloads
        enable_device_map = os.getenv("HUNYUAN_PAINT_DEVICE_MAP", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
        enable_disk_offload = os.getenv("HUNYUAN_PAINT_DISK_OFFLOAD", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

        def _host_mem_summary() -> str:
            try:
                import psutil

                vm = psutil.virtual_memory()
                rss = int(psutil.Process(os.getpid()).memory_info().rss)
                return (
                    f"host_rss={rss / 1024**3:.2f}GiB "
                    f"host_free={int(vm.available) / 1024**3:.2f}GiB "
                    f"host_total={int(vm.total) / 1024**3:.2f}GiB"
                )
            except Exception:
                return "host=unknown"

        def _log(message: str) -> None:
            if os.getenv("HUNYUAN_MEMORY_DEBUG", "0").strip().lower() in {"1", "true", "yes", "y", "on"}:
                print(f"[hunyuan-paint] {message} {_host_mem_summary()}", flush=True)

        def _read_gpu_reserve_bytes() -> Tuple[float, int]:
            reserve_frac_raw = os.getenv("GPU_RESERVE_FRAC")
            if reserve_frac_raw is None or reserve_frac_raw.strip() == "":
                reserve_frac = 0.15
            else:
                try:
                    reserve_frac = float(reserve_frac_raw)
                except ValueError as exc:
                    raise RuntimeError(f"GPU_RESERVE_FRAC must be a number, got {reserve_frac_raw!r}") from exc
                if reserve_frac < 0.0 or reserve_frac > 0.90:
                    raise RuntimeError(f"GPU_RESERVE_FRAC must be in [0, 0.90], got {reserve_frac}")

            reserve_floor_bytes_default = int(1.5 * 1024**3)
            reserve_floor_bytes_raw = os.getenv("GPU_RESERVE_FLOOR_BYTES")
            reserve_floor_bytes = reserve_floor_bytes_default
            if reserve_floor_bytes_raw is not None and reserve_floor_bytes_raw.strip() != "":
                try:
                    reserve_floor_bytes = int(reserve_floor_bytes_raw)
                except ValueError as exc:
                    raise RuntimeError(
                        f"GPU_RESERVE_FLOOR_BYTES must be an integer, got {reserve_floor_bytes_raw!r}"
                    ) from exc
                if reserve_floor_bytes < 0:
                    raise RuntimeError(f"GPU_RESERVE_FLOOR_BYTES must be >= 0, got {reserve_floor_bytes}")
            else:
                reserve_floor_gb_raw = os.getenv("GPU_RESERVE_FLOOR_GB")
                if reserve_floor_gb_raw is not None and reserve_floor_gb_raw.strip() != "":
                    try:
                        reserve_floor_gb = float(reserve_floor_gb_raw)
                    except ValueError as exc:
                        raise RuntimeError(f"GPU_RESERVE_FLOOR_GB must be a number, got {reserve_floor_gb_raw!r}") from exc
                    if reserve_floor_gb < 0:
                        raise RuntimeError(f"GPU_RESERVE_FLOOR_GB must be >= 0, got {reserve_floor_gb}")
                    reserve_floor_bytes = int(reserve_floor_gb * 1e9)

            return reserve_frac, reserve_floor_bytes

        def _gpu_headroom_bytes() -> Optional[Tuple[int, int, int]]:
            if not torch.cuda.is_available():
                return None
            try:
                cuda_device = torch.device(self.device)
            except Exception:
                return None
            if cuda_device.type != "cuda":
                return None
            device_index = cuda_device.index
            if device_index is None:
                device_index = int(torch.cuda.current_device())
            if device_index < 0 or device_index >= int(torch.cuda.device_count()):
                raise RuntimeError(
                    f"Invalid CUDA device {self.device!r}; visible device_count={torch.cuda.device_count()}. "
                    "Set config.device to a valid 'cuda:N' or adjust CUDA_VISIBLE_DEVICES."
                )
            with torch.cuda.device(device_index):
                free_bytes, total_bytes = torch.cuda.mem_get_info()
            reserve_frac, reserve_floor_bytes = _read_gpu_reserve_bytes()
            reserve_target_bytes = max(int(total_bytes * reserve_frac), reserve_floor_bytes)
            headroom_bytes = max(0, int(free_bytes) - reserve_target_bytes)
            return int(total_bytes), int(free_bytes), int(headroom_bytes)

        def _ensure_offload_dir() -> Optional[str]:
            if not enable_disk_offload:
                return None
            offload_dir = os.getenv("HUNYUAN_PAINT_OFFLOAD_DIR", "/volumes/weights/hunyuan-offload/paint").strip()
            if offload_dir == "":
                return None
            try:
                os.makedirs(offload_dir, exist_ok=True)
            except OSError as exc:
                raise RuntimeError(
                    f"HUNYUAN_PAINT_OFFLOAD_DIR={offload_dir!r} is not writable. "
                    "Set it to a writable, preferably NVMe-backed path."
                ) from exc
            return offload_dir

        def _sum_weight_files(component_dir: str) -> Optional[int]:
            try:
                entries = os.listdir(component_dir)
            except OSError:
                return None
            weight_bytes = 0
            found = False
            for entry in entries:
                if entry.endswith(".safetensors") or entry.endswith(".bin"):
                    lowered = entry.lower()
                    if "optimizer" in lowered or "scheduler" in lowered:
                        continue
                    try:
                        weight_bytes += os.path.getsize(os.path.join(component_dir, entry))
                        found = True
                    except OSError:
                        continue
            return weight_bytes if found else None

        def _estimate_component_weights(model_root: str) -> Dict[str, int]:
            index_path = os.path.join(model_root, "model_index.json")
            components: Dict[str, int] = {}
            if os.path.exists(index_path):
                try:
                    with open(index_path, "r", encoding="utf-8") as handle:
                        index = json.load(handle)
                except Exception:
                    index = {}
                for key in index:
                    if key.startswith("_"):
                        continue
                    component_dir = os.path.join(model_root, key)
                    if not os.path.isdir(component_dir):
                        continue
                    weight_bytes = _sum_weight_files(component_dir)
                    if weight_bytes is not None:
                        components[key] = weight_bytes

            if components:
                return components

            fallback_keys = ("unet", "vae", "text_encoder", "image_encoder")
            for key in fallback_keys:
                component_dir = os.path.join(model_root, key)
                if os.path.isdir(component_dir):
                    weight_bytes = _sum_weight_files(component_dir)
                    if weight_bytes is not None:
                        components[key] = weight_bytes
            return components

        def _priority(component_name: str) -> int:
            if component_name == "unet":
                return 100
            if component_name == "vae":
                return 90
            if component_name == "text_encoder":
                return 80
            if component_name == "image_encoder":
                return 70
            if "unet" in component_name:
                return 60
            if "encoder" in component_name:
                return 50
            return 10

        def _plan_device_map(model_root: str) -> Optional[Dict[str, str]]:
            if not enable_device_map:
                return None
            if not torch.cuda.is_available():
                return None
            if not isinstance(self.device, str) or not self.device.startswith("cuda"):
                return None

            gpu_state = _gpu_headroom_bytes()
            if gpu_state is None:
                return None
            total_bytes, free_bytes, headroom_bytes = gpu_state
            components = _estimate_component_weights(model_root)
            if not components:
                return None

            pinned: set[str] = set()
            remaining = headroom_bytes
            sorted_components = sorted(components.items(), key=lambda item: (_priority(item[0]), item[1]), reverse=True)
            for name, weight_bytes in sorted_components:
                if name == "unet":
                    pinned.add(name)
                    remaining -= weight_bytes
                    continue
                if weight_bytes <= remaining:
                    pinned.add(name)
                    remaining -= weight_bytes

            offload_targets = {name for name in components.keys() if name not in pinned}
            if enable_disk_offload:
                device_map = {name: (self.device if name in pinned else "disk") for name in components.keys()}
            else:
                device_map = {name: (self.device if name in pinned else "cpu") for name in components.keys()}

            _log(
                "device_map planned "
                f"gpu_total={total_bytes / 1024**3:.1f}GiB gpu_free={free_bytes / 1024**3:.1f}GiB "
                f"gpu_headroom={headroom_bytes / 1024**3:.1f}GiB "
                f"pinned={sorted(pinned)} offloaded={sorted(offload_targets)}"
            )
            return device_map

        cfg_path = config.multiview_cfg_path
        custom_pipeline = os.path.join(os.path.dirname(__file__),"..","hunyuanpaintpbr")
        cfg = OmegaConf.load(cfg_path)
        self.cfg = cfg
        self.mode = self.cfg.model.params.stable_diffusion_config.custom_pipeline[2:]

        try:
            _log(
                f"snapshot_download repo={config.multiview_pretrained_path} "
                f"allow_patterns=hunyuan3d-paintpbr-v2-1/* local_files_only={local_files_only}"
            )
            model_path = huggingface_hub.snapshot_download(
                repo_id=config.multiview_pretrained_path,
                allow_patterns=["hunyuan3d-paintpbr-v2-1/*"],
                local_files_only=local_files_only,
            )
        except Exception as exc:
            if local_files_only:
                raise RuntimeError(
                    "Paint model weights are not present in the local HuggingFace cache. "
                    "Run the prefetch step (populate weights volume) or set HUNYUAN_HF_ALLOW_DOWNLOADS=1."
                ) from exc
            raise

        model_path = os.path.join(model_path, "hunyuan3d-paintpbr-v2-1")
        from_pretrained_kwargs = {
            "custom_pipeline": custom_pipeline,
            "torch_dtype": torch.float16,
            # Avoid large peak RSS on small hosts while loading multi-GB .bin weights.
            "low_cpu_mem_usage": True,
        }
        if local_files_only:
            from_pretrained_kwargs["local_files_only"] = True
        offload_dir = _ensure_offload_dir()
        if offload_dir is not None:
            from_pretrained_kwargs["offload_folder"] = offload_dir
            from_pretrained_kwargs["offload_state_dict"] = True
        planned_device_map = _plan_device_map(model_path)
        if planned_device_map is not None:
            from_pretrained_kwargs["device_map"] = planned_device_map
        try:
            import transformers

            clip_from_pretrained = transformers.CLIPTextModel.from_pretrained
            vision_from_pretrained = transformers.CLIPVisionModelWithProjection.from_pretrained
            tok_from_pretrained = transformers.CLIPTokenizer.from_pretrained

            def _clip_from_pretrained_wrapped(*args, **kwargs):
                kwargs.setdefault("torch_dtype", torch.float16)
                kwargs.setdefault("low_cpu_mem_usage", True)
                kwargs.setdefault("local_files_only", local_files_only)
                if offload_dir is not None:
                    kwargs.setdefault("offload_folder", offload_dir)
                    kwargs.setdefault("offload_state_dict", True)
                start = time.monotonic()
                _log(f"CLIPTextModel.from_pretrained start local_files_only={local_files_only}")
                try:
                    model = clip_from_pretrained(*args, **kwargs)
                except TypeError:
                    kwargs.pop("offload_folder", None)
                    kwargs.pop("offload_state_dict", None)
                    model = clip_from_pretrained(*args, **kwargs)
                _log(f"CLIPTextModel.from_pretrained done elapsed={time.monotonic() - start:.1f}s")
                return model

            def _vision_from_pretrained_wrapped(*args, **kwargs):
                kwargs.setdefault("torch_dtype", torch.float16)
                kwargs.setdefault("low_cpu_mem_usage", True)
                kwargs.setdefault("local_files_only", local_files_only)
                if offload_dir is not None:
                    kwargs.setdefault("offload_folder", offload_dir)
                    kwargs.setdefault("offload_state_dict", True)
                start = time.monotonic()
                _log(f"CLIPVisionModelWithProjection.from_pretrained start local_files_only={local_files_only}")
                try:
                    model = vision_from_pretrained(*args, **kwargs)
                except TypeError:
                    kwargs.pop("offload_folder", None)
                    kwargs.pop("offload_state_dict", None)
                    model = vision_from_pretrained(*args, **kwargs)
                _log(f"CLIPVisionModelWithProjection.from_pretrained done elapsed={time.monotonic() - start:.1f}s")
                return model

            def _tok_from_pretrained_wrapped(*args, **kwargs):
                kwargs.setdefault("local_files_only", local_files_only)
                return tok_from_pretrained(*args, **kwargs)

            transformers.CLIPTextModel.from_pretrained = _clip_from_pretrained_wrapped
            transformers.CLIPVisionModelWithProjection.from_pretrained = _vision_from_pretrained_wrapped
            transformers.CLIPTokenizer.from_pretrained = _tok_from_pretrained_wrapped

            try:
                _log("DiffusionPipeline.from_pretrained start")
                try:
                    pipeline = DiffusionPipeline.from_pretrained(model_path, **from_pretrained_kwargs)
                except torch.cuda.OutOfMemoryError:
                    if planned_device_map is None:
                        raise
                    _log(
                        "DiffusionPipeline.from_pretrained OOM with planned device_map; "
                        "retrying with unet pinned and remaining weights offloaded"
                    )
                    fallback_kwargs = dict(from_pretrained_kwargs)
                    fallback_device_map = {}
                    for name in planned_device_map.keys():
                        if name == "unet":
                            fallback_device_map[name] = self.device
                        else:
                            fallback_device_map[name] = "disk" if enable_disk_offload else "cpu"
                    fallback_kwargs["device_map"] = fallback_device_map
                    pipeline = DiffusionPipeline.from_pretrained(model_path, **fallback_kwargs)
                _log("DiffusionPipeline.from_pretrained done")
            finally:
                transformers.CLIPTextModel.from_pretrained = clip_from_pretrained
                transformers.CLIPVisionModelWithProjection.from_pretrained = vision_from_pretrained
                transformers.CLIPTokenizer.from_pretrained = tok_from_pretrained
        except TypeError:
            from_pretrained_kwargs.pop("low_cpu_mem_usage", None)
            from_pretrained_kwargs.pop("device_map", None)
            from_pretrained_kwargs.pop("offload_folder", None)
            from_pretrained_kwargs.pop("offload_state_dict", None)
            _log("DiffusionPipeline.from_pretrained retry without low_cpu_mem_usage")
            pipeline = DiffusionPipeline.from_pretrained(model_path, **from_pretrained_kwargs)

        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
        pipeline.set_progress_bar_config(disable=True)
        pipeline.eval()
        setattr(pipeline, "view_size", cfg.model.params.get("view_size", 320))
        if planned_device_map is None:
            self.pipeline = pipeline.to(self.device)
        else:
            self.pipeline = pipeline

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            from hunyuanpaintpbr.unet.modules import Dino_v2
            self.dino_v2 = Dino_v2(config.dino_ckpt_path).to(torch.float16)
            self.dino_v2 = self.dino_v2.to(self.device)

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PL_GLOBAL_SEED"] = str(seed)

    @torch.no_grad()
    def __call__(self, images, conditions, prompt=None, custom_view_size=None, resize_input=False):
        pils = self.forward_one(
            images, conditions, prompt=prompt, custom_view_size=custom_view_size, resize_input=resize_input
        )
        return pils

    def forward_one(self, input_images, control_images, prompt=None, custom_view_size=None, resize_input=False):
        self.seed_everything(0)
        custom_view_size = custom_view_size if custom_view_size is not None else self.pipeline.view_size
        if not isinstance(input_images, List):
            input_images = [input_images]
        if not resize_input:
            input_images = [
                input_image.resize((self.pipeline.view_size, self.pipeline.view_size)) for input_image in input_images
            ]
        else:
            input_images = [input_image.resize((custom_view_size, custom_view_size)) for input_image in input_images]
        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((custom_view_size, custom_view_size))
            if control_images[i].mode == "L":
                control_images[i] = control_images[i].point(lambda x: 255 if x > 1 else 0, mode="1")
        generator_device = self.pipeline.device
        try:
            generator_device = torch.device(self.device)
        except Exception:
            pass
        kwargs = dict(generator=torch.Generator(device=generator_device).manual_seed(0))

        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]

        kwargs["width"] = custom_view_size
        kwargs["height"] = custom_view_size
        kwargs["num_in_batch"] = num_view
        kwargs["images_normal"] = normal_image
        kwargs["images_position"] = position_image

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            dino_hidden_states = self.dino_v2(input_images[0])
            kwargs["dino_hidden_states"] = dino_hidden_states

        sync_condition = None

        infer_steps_dict = {
            "EulerAncestralDiscreteScheduler": 30,
            "UniPCMultistepScheduler": 15,
            "DDIMScheduler": 50,
            "ShiftSNRScheduler": 15,
        }

        mvd_image = self.pipeline(
            input_images[0:1],
            num_inference_steps=infer_steps_dict[self.pipeline.scheduler.__class__.__name__],
            prompt=prompt,
            sync_condition=sync_condition,
            guidance_scale=3.0,
            **kwargs,
        ).images

        if "pbr" in self.mode:
            mvd_image = {"albedo": mvd_image[:num_view], "mr": mvd_image[num_view:]}
            # mvd_image = {'albedo':mvd_image[:num_view]}
        else:
            mvd_image = {"hdr": mvd_image}

        return mvd_image
