from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


# 图像预处理的标准均值和标准差
IMG_MEAN = (0.5, 0.5, 0.5)
IMG_STD = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class TrackerBuildConfig:
    """追踪器构建配置，定义模型结构和内存参数。"""
    profile_name: str = "default"  # 配置方案名称
    image_size: int = 1008  # 输入图像尺寸
    num_maskmem: int = 7  # 掩码内存大小
    max_cond_frames_in_attn: int = 4  # 注意力机制中最大条件帧数
    max_obj_ptrs_in_encoder: int = 16  # 编码器中最大物体指针数
    multimask_output_in_sam: bool = True  # SAM 是否输出多掩码
    multimask_output_for_tracking: bool = True  # 追踪是否使用多掩码输出
    trim_past_non_cond_mem_for_eval: bool = False  # 评估时是否修剪过去的非条件内存
    use_memory_selection: bool = False  # 是否使用内存选择机制


def resolve_tracker_build_config(
    *,
    image_size_override: int | None = None,
) -> TrackerBuildConfig:
    """解析追踪器构建配置。"""
    config = TrackerBuildConfig()
    if image_size_override is not None:
        image_size_override = int(image_size_override)
        if image_size_override <= 0 or image_size_override % 14 != 0:
            raise ValueError("tracker image_size must be a positive multiple of 14")
        config = replace(config, image_size=int(image_size_override))
    return config


def _load_tracker_state_dict(
    checkpoint_path: Path,
    *,
    build_config: TrackerBuildConfig,
) -> dict[str, Any]:
    """从权重文件中加载追踪器的状态字典并进行适配。"""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    state_dict: dict[str, Any] = {}
    for key, value in ckpt.items():
        key_str = str(key)
        # 提取追踪器和主干网络相关的权重
        if key_str.startswith("tracker."):
            state_dict[key_str.replace("tracker.", "")] = value
        elif key_str.startswith("detector.backbone."):
            state_dict[key_str.replace("detector.backbone.", "backbone.")] = value
    
    # 根据构建配置调整权重形状
    state_dict = adapt_tracker_state_dict_for_config(state_dict, build_config=build_config)
    return state_dict


def adapt_tracker_state_dict_for_config(
    state_dict: dict[str, Any],
    *,
    build_config: TrackerBuildConfig,
) -> dict[str, Any]:
    """根据追踪器配置调整状态字典中的权重（例如内存时间位置编码）。"""
    adapted = dict(state_dict)
    tpos_key = "maskmem_tpos_enc"
    if tpos_key in adapted:
        maskmem_tpos_enc = adapted[tpos_key]
        if int(maskmem_tpos_enc.shape[0]) != int(build_config.num_maskmem):
            adapted[tpos_key] = maskmem_tpos_enc[: int(build_config.num_maskmem)].contiguous()
    return adapted


def filter_incompatible_state_dict_for_model(
    model: torch.nn.Module,
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """过滤掉与模型当前结构不兼容的状态字典条目。"""
    model_state = model.state_dict()
    compatible: dict[str, Any] = {}
    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            compatible[key] = value
            continue
        if hasattr(value, "shape") and hasattr(target, "shape") and tuple(value.shape) != tuple(target.shape):
            continue
        compatible[key] = value
    return compatible


def build_local_single_object_tracker(
    *,
    build_config: TrackerBuildConfig,
):
    from sam3.model.decoder import TransformerDecoderLayerv2, TransformerEncoderCrossAttention
    from sam3.model.memory import CXBlock, SimpleFuser, SimpleMaskDownSampler, SimpleMaskEncoder
    from sam3.model.model_misc import TransformerWrapper
    from sam3.model.necks import Sam3DualViTDetNeck
    from sam3.model.position_encoding import PositionEmbeddingSine
    from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
    from sam3.model.vitdet import ViT
    from sam3.model.vl_combiner import SAM3VLBackbone
    from sam3.sam.transformer import RoPEAttention

    image_size = int(build_config.image_size)
    feat_size = int(image_size // 14)

    vit_backbone = ViT(
        img_size=image_size,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=None,
    )
    vision_backbone = Sam3DualViTDetNeck(
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
            scale=None,
            temperature=10000,
            precompute_resolution=image_size,
        ),
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=True,
    )
    maskmem_backbone = SimpleMaskEncoder(
        out_dim=64,
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=64,
            normalize=True,
            scale=None,
            temperature=10000,
            precompute_resolution=image_size,
        ),
        mask_downsampler=SimpleMaskDownSampler(
            kernel_size=3,
            stride=2,
            padding=1,
            interpol_size=[int(feat_size * 16), int(feat_size * 16)],
        ),
        fuser=SimpleFuser(
            layer=CXBlock(
                dim=256,
                kernel_size=7,
                padding=3,
                layer_scale_init_value=1.0e-06,
                use_dwconv=True,
            ),
            num_layers=2,
        ),
    )
    self_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        rope_theta=10000.0,
        feat_sizes=[feat_size, feat_size],
        use_fa3=False,
        use_rope_real=False,
    )
    cross_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        kv_in_dim=64,
        rope_theta=10000.0,
        feat_sizes=[feat_size, feat_size],
        rope_k_repeat=True,
        use_fa3=False,
        use_rope_real=False,
    )
    transformer = TransformerWrapper(
        encoder=TransformerEncoderCrossAttention(
            remove_cross_attention_layers=[],
            batch_first=True,
            d_model=256,
            frozen=False,
            pos_enc_at_input=True,
            layer=TransformerDecoderLayerv2(
                cross_attention_first=False,
                activation="relu",
                dim_feedforward=2048,
                dropout=0.1,
                pos_enc_at_attn=False,
                pre_norm=True,
                self_attention=self_attention,
                d_model=256,
                pos_enc_at_cross_attn_keys=True,
                pos_enc_at_cross_attn_queries=False,
                cross_attention=cross_attention,
            ),
            num_layers=4,
            use_act_checkpoint=False,
        ),
        decoder=None,
        d_model=256,
    )
    backbone = SAM3VLBackbone(scalp=1, visual=vision_backbone, text=None)
    return Sam3TrackerPredictor(
        image_size=image_size,
        num_maskmem=int(build_config.num_maskmem),
        backbone=backbone,
        backbone_stride=14,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        multimask_output_in_sam=bool(build_config.multimask_output_in_sam),
        forward_backbone_per_frame_for_eval=True,
        trim_past_non_cond_mem_for_eval=bool(build_config.trim_past_non_cond_mem_for_eval),
        multimask_output_for_tracking=bool(build_config.multimask_output_for_tracking),
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        always_start_from_first_ann_frame=False,
        non_overlap_masks_for_mem_enc=False,
        non_overlap_masks_for_output=False,
        max_cond_frames_in_attn=int(build_config.max_cond_frames_in_attn),
        max_obj_ptrs_in_encoder=int(build_config.max_obj_ptrs_in_encoder),
        offload_output_to_cpu_for_eval=False,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        clear_non_cond_mem_around_input=True,
        fill_hole_area=0,
        use_memory_selection=bool(build_config.use_memory_selection),
    )


def build_single_object_tracker_model(
    checkpoint_path: Path,
    *,
    device: str = "cuda",
    image_size_override: int | None = None,
):
    build_config = resolve_tracker_build_config(
        image_size_override=image_size_override,
    )
    model = build_local_single_object_tracker(
        build_config=build_config,
    )
    state_dict = _load_tracker_state_dict(
        Path(checkpoint_path),
        build_config=build_config,
    )
    state_dict = filter_incompatible_state_dict_for_model(model, state_dict)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if device == "cuda":
        model = model.cuda()
    model.eval()
    model._single_seg_missing_keys = tuple(missing_keys)
    model._single_seg_unexpected_keys = tuple(unexpected_keys)
    return model, missing_keys


def preprocess_pil_image(
    image: Image.Image | np.ndarray,
    *,
    image_size: int,
    device: torch.device,
    img_mean: torch.Tensor | None = None,
    img_std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, int]:
    if isinstance(image, Image.Image):
        image_rgb = image if image.mode == "RGB" else image.convert("RGB")
        video_width, video_height = image_rgb.size
        image_np = np.array(image_rgb, dtype=np.uint8, copy=True)
    else:
        image_np = np.asarray(image, dtype=np.uint8)
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 uint8 image array, got shape {image_np.shape}")
        if not image_np.flags.c_contiguous:
            image_np = np.ascontiguousarray(image_np)
        video_height, video_width = int(image_np.shape[0]), int(image_np.shape[1])
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device, non_blocking=True)
    tensor = tensor.to(dtype=torch.float32).div_(255.0)
    tensor = F.interpolate(
        tensor,
        size=(int(image_size), int(image_size)),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    if img_mean is None:
        img_mean = torch.tensor(IMG_MEAN, dtype=torch.float32, device=device)[:, None, None]
    if img_std is None:
        img_std = torch.tensor(IMG_STD, dtype=torch.float32, device=device)[:, None, None]
    tensor.sub_(img_mean).div_(img_std)
    return tensor.squeeze(0), int(video_height), int(video_width)


@dataclass
class TrackerOnlySession:
    session_id: str
    state: dict[str, Any]
    needs_preflight: bool = False


@dataclass(frozen=True)
class StitchedTile:
    camera_id: str
    x: int
    y: int
    width: int
    height: int


@dataclass
class StitchedLayout:
    canvas_width: int
    canvas_height: int
    camera_ids: list[str]
    tiles: dict[str, StitchedTile]


@dataclass(frozen=True)
class CropWindow:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return int(self.x1 - self.x0)

    @property
    def height(self) -> int:
        return int(self.y1 - self.y0)


def build_stitched_layout(
    frame_sizes: dict[str, tuple[int, int]],
    camera_order: list[str],
) -> StitchedLayout:
    if not camera_order:
        raise ValueError("camera_order must not be empty")
    cell_width = max(int(frame_sizes[camera_id][0]) for camera_id in camera_order)
    cell_height = max(int(frame_sizes[camera_id][1]) for camera_id in camera_order)
    cols = 1 if len(camera_order) == 1 else 2
    rows = int(np.ceil(len(camera_order) / cols))
    tiles: dict[str, StitchedTile] = {}
    for index, camera_id in enumerate(camera_order):
        col = index % cols
        row = index // cols
        width, height = frame_sizes[camera_id]
        tiles[camera_id] = StitchedTile(
            camera_id=str(camera_id),
            x=int(col * cell_width),
            y=int(row * cell_height),
            width=int(width),
            height=int(height),
        )
    return StitchedLayout(
        canvas_width=int(cols * cell_width),
        canvas_height=int(rows * cell_height),
        camera_ids=[str(camera_id) for camera_id in camera_order],
        tiles=tiles,
    )


def compose_camera_frame_resources(
    frame_resources: dict[str, list[Image.Image]],
    camera_order: list[str],
    layout: StitchedLayout | None = None,
) -> tuple[list[Image.Image], StitchedLayout]:
    frame_sizes = {
        camera_id: tuple(int(value) for value in frame_resources[camera_id][0].size)
        for camera_id in camera_order
    }
    if layout is None:
        layout = build_stitched_layout(frame_sizes, camera_order)
    else:
        if [str(camera_id) for camera_id in camera_order] != list(layout.camera_ids):
            raise ValueError("camera_order does not match stitched layout camera_ids")
        for camera_id in camera_order:
            expected = layout.tiles[camera_id]
            width, height = frame_sizes[camera_id]
            if int(width) != int(expected.width) or int(height) != int(expected.height):
                raise ValueError(f"frame size mismatch for {camera_id}: {(width, height)} vs {(expected.width, expected.height)}")
    canvas = Image.new("RGB", (int(layout.canvas_width), int(layout.canvas_height)), (0, 0, 0))
    for camera_id in camera_order:
        tile = layout.tiles[camera_id]
        canvas.paste(frame_resources[camera_id][0], (int(tile.x), int(tile.y)))
    return [canvas], layout


def compose_camera_rgb_frame_resources(
    rgb_by_camera: dict[str, np.ndarray],
    camera_order: list[str],
    layout: StitchedLayout | None = None,
) -> tuple[list[np.ndarray], StitchedLayout]:
    frame_sizes = {
        camera_id: (int(rgb_by_camera[camera_id].shape[1]), int(rgb_by_camera[camera_id].shape[0]))
        for camera_id in camera_order
    }
    if layout is None:
        layout = build_stitched_layout(frame_sizes, camera_order)
    else:
        if [str(camera_id) for camera_id in camera_order] != list(layout.camera_ids):
            raise ValueError("camera_order does not match stitched layout camera_ids")
        for camera_id in camera_order:
            expected = layout.tiles[camera_id]
            width, height = frame_sizes[camera_id]
            if int(width) != int(expected.width) or int(height) != int(expected.height):
                raise ValueError(f"frame size mismatch for {camera_id}: {(width, height)} vs {(expected.width, expected.height)}")
    canvas = np.zeros((int(layout.canvas_height), int(layout.canvas_width), 3), dtype=np.uint8)
    for camera_id in camera_order:
        tile = layout.tiles[camera_id]
        rgb = np.asarray(rgb_by_camera[camera_id], dtype=np.uint8)
        if rgb.shape[:2] != (int(tile.height), int(tile.width)):
            raise ValueError(f"rgb shape mismatch for {camera_id}: {rgb.shape[:2]} vs {(tile.height, tile.width)}")
        canvas[int(tile.y) : int(tile.y + tile.height), int(tile.x) : int(tile.x + tile.width)] = rgb
    return [canvas], layout


def stitch_camera_binary_masks(
    masks_by_camera: dict[str, np.ndarray],
    layout: StitchedLayout,
) -> np.ndarray:
    canvas = np.zeros((int(layout.canvas_height), int(layout.canvas_width)), dtype=bool)
    for camera_id in layout.camera_ids:
        if camera_id not in masks_by_camera:
            continue
        tile = layout.tiles[camera_id]
        mask = np.asarray(masks_by_camera[camera_id], dtype=bool)
        if mask.shape != (int(tile.height), int(tile.width)):
            raise ValueError(f"mask shape mismatch for {camera_id}: {mask.shape} vs {(tile.height, tile.width)}")
        canvas[int(tile.y) : int(tile.y + tile.height), int(tile.x) : int(tile.x + tile.width)] = mask
    return canvas


def split_stitched_binary_mask(
    composite_mask: np.ndarray,
    layout: StitchedLayout,
) -> dict[str, np.ndarray]:
    composite_bool = np.asarray(composite_mask, dtype=bool)
    if composite_bool.shape != (int(layout.canvas_height), int(layout.canvas_width)):
        raise ValueError(
            f"composite mask shape mismatch: {composite_bool.shape} vs {(layout.canvas_height, layout.canvas_width)}"
        )
    return {
        camera_id: composite_bool[
            int(tile.y) : int(tile.y + tile.height),
            int(tile.x) : int(tile.x + tile.width),
        ]
        for camera_id, tile in layout.tiles.items()
    }


def split_stitched_binary_mask_torch(
    composite_mask: torch.Tensor,
    layout: StitchedLayout,
) -> dict[str, torch.Tensor]:
    composite_bool = torch.as_tensor(composite_mask, dtype=torch.bool)
    if tuple(composite_bool.shape) != (int(layout.canvas_height), int(layout.canvas_width)):
        raise ValueError(
            f"composite mask shape mismatch: {tuple(composite_bool.shape)} vs {(layout.canvas_height, layout.canvas_width)}"
        )
    return {
        camera_id: composite_bool[
            int(tile.y) : int(tile.y + tile.height),
            int(tile.x) : int(tile.x + tile.width),
        ]
        for camera_id, tile in layout.tiles.items()
    }


def _resize_binary_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    mask_uint8 = (np.asarray(mask, dtype=bool).astype(np.uint8) * 255)
    resized = Image.fromarray(mask_uint8, mode="L").resize(
        (int(size[0]), int(size[1])),
        resample=Image.Resampling.NEAREST,
    )
    return np.asarray(resized, dtype=np.uint8) > 0


def full_frame_crop_window(image_size: tuple[int, int]) -> CropWindow:
    width, height = (int(image_size[0]), int(image_size[1]))
    return CropWindow(x0=0, y0=0, x1=width, y1=height)


def crop_window_from_mask(
    mask: np.ndarray,
    *,
    image_size: tuple[int, int],
    margin_scale: float = 2.0,
    min_size_ratio: float = 0.35,
) -> CropWindow:
    width, height = (int(image_size[0]), int(image_size[1]))
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != (height, width):
        raise ValueError(f"mask shape mismatch: {mask_bool.shape} vs {(height, width)}")
    ys, xs = np.nonzero(mask_bool)
    if ys.size == 0 or xs.size == 0:
        return full_frame_crop_window((width, height))
    x_min = int(xs.min())
    x_max = int(xs.max()) + 1
    y_min = int(ys.min())
    y_max = int(ys.max()) + 1
    box_width = max(1, x_max - x_min)
    box_height = max(1, y_max - y_min)
    min_crop_width = max(1, int(round(width * float(min_size_ratio))))
    min_crop_height = max(1, int(round(height * float(min_size_ratio))))
    crop_width = min(width, max(min_crop_width, int(round(box_width * float(margin_scale)))))
    crop_height = min(height, max(min_crop_height, int(round(box_height * float(margin_scale)))))
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    x0 = int(round(center_x - crop_width / 2.0))
    y0 = int(round(center_y - crop_height / 2.0))
    x0 = max(0, min(x0, width - crop_width))
    y0 = max(0, min(y0, height - crop_height))
    return CropWindow(
        x0=int(x0),
        y0=int(y0),
        x1=int(x0 + crop_width),
        y1=int(y0 + crop_height),
    )


def crop_and_resize_frame(
    image: Image.Image,
    crop_window: CropWindow,
    *,
    output_size: tuple[int, int],
) -> Image.Image:
    cropped = image.crop((int(crop_window.x0), int(crop_window.y0), int(crop_window.x1), int(crop_window.y1)))
    return cropped.resize((int(output_size[0]), int(output_size[1])), resample=Image.Resampling.BILINEAR)


def crop_mask_to_tracker_view(
    mask: np.ndarray,
    crop_window: CropWindow,
    *,
    output_size: tuple[int, int],
) -> np.ndarray:
    cropped = np.asarray(mask, dtype=bool)[
        int(crop_window.y0) : int(crop_window.y1),
        int(crop_window.x0) : int(crop_window.x1),
    ]
    return _resize_binary_mask(cropped, output_size)


def project_tracker_mask_to_full_image(
    tracker_mask: np.ndarray,
    crop_window: CropWindow,
    *,
    full_size: tuple[int, int],
) -> np.ndarray:
    full_width, full_height = (int(full_size[0]), int(full_size[1]))
    resized = _resize_binary_mask(tracker_mask, (int(crop_window.width), int(crop_window.height)))
    canvas = np.zeros((full_height, full_width), dtype=bool)
    canvas[int(crop_window.y0) : int(crop_window.y1), int(crop_window.x0) : int(crop_window.x1)] = resized
    return canvas


class TrackerOnlyVideoPredictor:
    """
    Thin local wrapper around Sam3TrackerPredictor with online frame append support.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        tracker_image_size: int | None = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("TrackerOnlyVideoPredictor requires CUDA")
        self.device = torch.device("cuda")
        self.model, self.missing_keys = build_single_object_tracker_model(
            checkpoint_path=Path(checkpoint_path),
            device="cuda",
            image_size_override=tracker_image_size,
        )
        self.image_size = int(self.model.image_size)
        self.sessions: dict[str, TrackerOnlySession] = {}
        self.img_mean = torch.tensor(IMG_MEAN, dtype=torch.float32, device=self.device)[:, None, None]
        self.img_std = torch.tensor(IMG_STD, dtype=torch.float32, device=self.device)[:, None, None]

    def _ensure_frame_list(self, resource_path) -> list[Image.Image]:
        if isinstance(resource_path, list):
            frames = resource_path
        else:
            frames = [resource_path]
        if not frames:
            raise TypeError("resource_path must not be empty")
        for frame in frames:
            if isinstance(frame, Image.Image):
                continue
            if isinstance(frame, np.ndarray):
                continue
            raise TypeError("resource_path must be a PIL.Image, np.ndarray, or a homogeneous list of them")
        return frames

    def _create_state_from_frames(self, frames: list[Image.Image]) -> dict[str, Any]:
        tensors: list[torch.Tensor] = []
        video_height: int | None = None
        video_width: int | None = None
        for frame in frames:
            tensor, frame_h, frame_w = preprocess_pil_image(
                frame,
                image_size=self.image_size,
                device=self.device,
                img_mean=self.img_mean,
                img_std=self.img_std,
            )
            tensors.append(tensor)
            if video_height is None:
                video_height = frame_h
                video_width = frame_w
        state = self.model.init_state(
            video_height=int(video_height),
            video_width=int(video_width),
            num_frames=len(tensors),
            cached_features=None,
        )
        state["images"] = tensors
        state["num_frames"] = len(tensors)
        return state

    def start_session(self, resource_path, session_id: str | None = None) -> dict[str, Any]:
        frames = self._ensure_frame_list(resource_path)
        session_id = str(session_id or uuid.uuid4())
        state = self._create_state_from_frames(frames)
        self.sessions[session_id] = TrackerOnlySession(session_id=session_id, state=state, needs_preflight=False)
        return {"session_id": session_id}

    def close_session(self, session_id: str) -> dict[str, Any]:
        self.sessions.pop(str(session_id), None)
        return {"is_success": True}

    def _get_session(self, session_id: str) -> TrackerOnlySession:
        try:
            return self.sessions[str(session_id)]
        except KeyError as exc:
            raise RuntimeError(f"Unknown tracker-only session: {session_id}") from exc

    def add_prompt(
        self,
        session_id: str,
        frame_idx: int,
        *,
        mask,
        obj_id: int,
        **_: Any,
    ) -> dict[str, Any]:
        session = self._get_session(session_id)
        mask_tensor = torch.as_tensor(mask, dtype=torch.float32)
        frame_idx_out, obj_ids, low_res_masks, video_res_masks = self.model.add_new_mask(
            inference_state=session.state,
            frame_idx=int(frame_idx),
            obj_id=int(obj_id),
            mask=mask_tensor,
        )
        session.needs_preflight = True
        outputs = self._pack_outputs(obj_ids, video_res_masks, None)
        return {"frame_index": int(frame_idx_out), "outputs": outputs}

    def append_frame(self, session_id: str, resource_path) -> dict[str, Any]:
        session = self._get_session(session_id)
        frames = self._ensure_frame_list(resource_path)
        if len(frames) != 1:
            raise RuntimeError("TrackerOnlyVideoPredictor.append_frame expects exactly one frame")
        tensor, _, _ = preprocess_pil_image(
            frames[0],
            image_size=self.image_size,
            device=self.device,
            img_mean=self.img_mean,
            img_std=self.img_std,
        )
        session.state["images"].append(tensor)
        session.state["num_frames"] = len(session.state["images"])
        return {"frame_index": int(session.state["num_frames"] - 1), "num_frames": int(session.state["num_frames"])}

    def _pack_outputs(self, obj_ids, video_res_masks, object_scores) -> dict[str, Any]:
        obj_ids_np = np.asarray(list(obj_ids), dtype=np.int32)
        if video_res_masks is None:
            return {
                "out_obj_ids": obj_ids_np,
                "out_binary_masks": np.zeros((0, 0, 0), dtype=bool),
                "out_mask_logits": np.zeros((0, 0, 0), dtype=np.float32),
                "out_probs": np.zeros((0,), dtype=np.float32),
                "out_tracker_probs": np.zeros((0,), dtype=np.float32),
            }
        mask_logits = video_res_masks.squeeze(1)
        binary_masks = mask_logits > 0
        if object_scores is None:
            probs = torch.ones(binary_masks.shape[0], dtype=torch.float32, device=binary_masks.device)
        else:
            probs = torch.sigmoid(object_scores.squeeze(-1).to(torch.float32))
        return {
            "out_obj_ids": obj_ids_np,
            "out_binary_masks": binary_masks,
            "out_mask_logits": mask_logits,
            "out_probs": probs,
            "out_tracker_probs": probs,
        }

    def infer_frame(self, session_id: str, frame_idx: int, reverse: bool = False) -> dict[str, Any]:
        session = self._get_session(session_id)
        payload = None
        for out_frame_idx, obj_ids, low_res_masks, video_res_masks, object_scores in self.model.propagate_in_video(
            inference_state=session.state,
            start_frame_idx=int(frame_idx),
            max_frame_num_to_track=0,
            reverse=bool(reverse),
            tqdm_disable=True,
            propagate_preflight=bool(session.needs_preflight),
        ):
            payload = {
                "frame_index": int(out_frame_idx),
                "outputs": self._pack_outputs(obj_ids, video_res_masks, object_scores),
            }
            break
        session.needs_preflight = False
        if payload is None:
            return {"frame_index": int(frame_idx), "outputs": None}
        return payload

    def infer_frames_batch(self, requests) -> list[dict[str, Any]]:
        return [
            self.infer_frame(
                session_id=request["session_id"],
                frame_idx=int(request["frame_index"]),
                reverse=bool(request.get("reverse", False)),
            )
            for request in requests
        ]
