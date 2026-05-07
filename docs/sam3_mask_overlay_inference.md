# SAM3 Mask Overlay Inference

This note documents the optional SAM3 mask-overlay path for testing policies trained with offline mask-overlaid images.

## Goal

During robot rollout, the client sends a normal OpenPI observation plus an optional `__mask_overlay` request. The policy server uses SAM3 to segment the target object, overlays the mask onto the selected camera image, and then runs the policy on that overlaid image.

The first request can contain a clicked point prompt. The server returns an overlay preview for confirmation. After confirmation, rollout requests omit the point prompt. The default `image` tracking mode reuses the previous mask/logits/bounding box as the next image prompt. The optional `video_window` tracking mode uses SAM3's video predictor over a short rolling frame window to propagate the initially selected object to the latest frame.

The overlay server is image-key agnostic: the requested `view` only needs to exist in `obs["image"]`. Current rollout clients use `front` for single-arm policies and `robot_0` / `robot_1` for dual-arm policies.

## Repository Setup

SAM3 is included as a submodule:

```bash
git submodule update --init --recursive third_party/sam3
```

SAM3 is only used when `scripts/serve_policy.py --mask-overlay` is enabled. In that mode the server imports SAM3, checks the checkpoint path, and loads the model before accepting client requests, so dependency or weight failures surface at startup instead of during the first rollout step.

SAM3 upstream currently requires a CUDA PyTorch environment and access to the checkpoint on Hugging Face. If `--sam3-checkpoint-path` is not provided, SAM3 will use its default Hugging Face download path, so make sure the server environment is authenticated.

## Start The Server

Example:

```bash
python scripts/serve_policy.py \
  --port 8009 \
  --mask-overlay \
  --mask-overlay-view robot_0 \
  --mask-overlay-alpha 0.35 \
  --mask-overlay-tracking-mode image \
  --sam3-checkpoint-path /path/to/sam3.pt \
  policy:checkpoint \
  --policy.config sponge_visual_guided_pi05 \
  --policy.dir /path/to/checkpoint/step

python scripts/serve_policy.py \
  --mask-overlay \
  --mask-overlay-view robot_0 \
  --mask-overlay-tracking-mode video_window \
  --mask-overlay-video-window-size 8 \
  --sam3-video-version sam3 \
  --sam3-checkpoint-path /root/Users/lixiaotong/openpi/third_party/sam3/ckpt/sam3.pt \
  policy:checkpoint \
  --policy.config sponge_visual_guided_pi05 \
  --policy.dir /root/Users/lixiaotong/openpi/checkpoints/sponge_visual_guided_pi05/sponge_visual_guided_pi05/34000
```

Useful server options:

- `--mask-overlay`: enables SAM3 overlay before policy inference.
- `--mask-overlay-view`: image key to overlay, for example `front`, `robot_0`, or `robot_1`.
- `--mask-overlay-alpha`: green overlay alpha.
- `--mask-overlay-tracking-mode`: `image` for previous bbox/logits image prompting, or `video_window` for rolling SAM3 video tracking.
- `--mask-overlay-video-window-size`: number of recent frames to rebuild as a video session in `video_window` mode.
- `--sam3-video-version`: SAM3 video predictor version, usually `sam3` for `sam3.pt` checkpoints.
- `--sam3-checkpoint-path`: optional local SAM3 checkpoint path.
- `--sam3-device`: SAM3 device, default `cuda`.
- `--sam3-path`: optional SAM3 source path; defaults to `third_party/sam3`.

Startup will fail fast if the local checkpoint path does not exist or if SAM3 cannot be imported. If `--sam3-checkpoint-path` is omitted, SAM3 may download its default checkpoint during server startup.

`video_window` mode loads both the SAM3 image interactivity model and the SAM3 video predictor. The image model is used only for the initial clicked-point segmentation; the resulting bbox becomes the visual prompt for the video tracker. This is more stable than repeatedly prompting the image model with a fixed point or previous bbox, but it is also slower and uses more GPU memory.

## Start Rollout With Click Prompt

Example:

```bash
python scripts/pi0_rollout_client_fasttouch_rpy.py \
  --description "..." \
  --arm_mode dual \
  --mask_overlay \
  --mask_view robot_0
```

The rollout script will:

1. Capture one resized frame from the selected camera view.
2. Open a click window if `--mask_prompt_point` is not provided.
3. Send the clicked point to the policy server with `preview_only=True`.
4. Display the server-returned mask overlay.
5. Continue rollout only after user confirmation.

To skip the click window:

```bash
--mask_prompt_point 120,90
```

Use `--mask_view robot_1` if the target object should be segmented from the second camera. For single-arm rollout, use `--arm_mode single --single_arm robot_0` or `--single_arm robot_1`; the client sends the selected camera as `image["front"]`, so `--mask_view` can be omitted and defaults to `front`.

Normal preview requests only return the overlay and mask. To debug black previews or transport issues, explicitly pass a debug directory:

```bash
--mask_debug_dir /tmp/openpi_mask_overlay_preview
```

With `--mask_debug_dir`, the client asks the server to echo the image it received during preview and normal rollout inference. Preview saves four debug files:

- `*_client.png`: image before websocket send.
- `*_server.png`: image received by the policy server.
- `*_overlay.png`: server-generated overlay.
- `*_mask.png`: server-generated mask.

During rollout, every policy inference also saves `rollout/<infer_index>_<view>_server.png`, `rollout/<infer_index>_<view>_overlay.png`, and `rollout/<infer_index>_<view>_mask.png`.

If the returned preview is black, compare these files first. If `*_client.png` is already black, the issue is camera capture or device mapping. If `*_client.png` is normal but `*_server.png` is black, the issue is serialization/input conversion. If both are normal but `*_overlay.png` is black, the issue is in server-side SAM3 overlay generation.

## Wire Protocol

The client adds a reserved key to the observation:

```python
obs["__mask_overlay"] = {
    "enabled": True,
    "view": "robot_0",
    "reset": True,
    "points": np.asarray([[120, 90]], dtype=np.float32),
    "point_labels": np.asarray([1], dtype=np.int32),
    "preview_only": True,
    "alpha": 0.35,
}
```

Preview response:

```python
resp["__mask_overlay"] = {
    "view": "robot_0",
    "tracking_mode": "image",
    "mask": mask_uint8,
    "overlay": overlay_rgb_uint8,
    "score": score_or_none,
    "bbox_xyxy": bbox_or_none,
    "initialized": True,
}
```

For normal rollout after confirmation, the client sends:

```python
obs["__mask_overlay"] = {
    "enabled": True,
    "view": "robot_0",
    "alpha": 0.35,
}
```

The server replaces `obs["image"][view]` with the overlay image before calling the policy, and attaches the latest overlay payload to the policy response for debugging.

## Implementation Notes

- The SAM3 wrapper lives in `src/openpi/serving/mask_overlay.py`.
- The server integration lives in `scripts/serve_policy.py`.
- The rollout click/preview flow lives in `scripts/pi0_rollout_client_fasttouch_rpy.py` and `scripts/pi0_rollout_client_xarm_rpy.py`.
- `image` mode uses SAM3 image interactivity with the previous mask/logits/bounding box as the next prompt.
- `video_window` mode uses SAM3's video predictor, but SAM3 initializes video state from a fixed image list/video path rather than an appendable stream. The server therefore rebuilds a short session from the latest frames for each policy inference, prompts frame 0 with the tracked bbox, and propagates forward to the newest frame.

## Troubleshooting

- If the client raises `server 未启用 --mask-overlay`, restart `serve_policy.py` with `--mask-overlay`.
- If SAM3 cannot download weights, authenticate Hugging Face or pass `--sam3-checkpoint-path`.
- If segmentation drifts in `image` mode, try `--mask-overlay-tracking-mode video_window`. If `video_window` drifts, reduce the window or reinitialize with a new click point.
- If `video_window` fails at startup, check that `--sam3-video-version` matches the checkpoint family. For `facebook/sam3` / `sam3.pt`, use `--sam3-video-version sam3`.
- If GPU memory is tight, run SAM3 and OpenPI on a larger GPU or use a separate server process/GPU for overlay experiments.
