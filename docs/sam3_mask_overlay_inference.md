# SAM3 Mask Overlay Inference

This note documents the optional SAM3 visual-guidance path for testing policies trained with either offline mask-overlaid images or separate mask image keys.

## Goal

During robot rollout, the client sends a normal OpenPI observation plus an optional `__mask_overlay` request. The policy server uses SAM3 to segment the target object, then either overlays the mask onto the selected camera image or injects the mask as a separate image key before running the policy.

The first request can contain a clicked point prompt. The server returns an overlay preview for confirmation. After confirmation, rollout requests omit the point prompt. The default `image` tracking mode reuses the previous mask/logits/bounding box as the next image prompt. The optional `video_window` tracking mode uses SAM3's video predictor over a short rolling frame window to propagate the initially selected object to the latest frame. The `text_select_video` mode first detects all instances matching a text prompt, selects the instance closest to the clicked point, and then keeps selecting the propagated instance that best matches the previous tracked bbox.

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
  --mask-overlay-view front \
  --mask-overlay-output-mode mask_image \
  --sam3-checkpoint-path /root/Users/lixiaotong/openpi/third_party/sam3/ckpt/sam3.pt \
  --mask-overlay-tracking-mode text_select_video \
  --mask-overlay-video-window-size 8 \
  --sam3-video-version sam3 \
  policy:checkpoint \
  --policy.config sponge_visual_mask_keys_pi05 \
  --policy.dir /root/Users/lixiaotong/openpi/checkpoints/sponge_visual_mask_keys_pi05/EXP/STEP
```

Useful server options:

- `--mask-overlay`: enables SAM3 overlay before policy inference.
- `--mask-overlay-view`: image key to overlay, for example `front`, `robot_0`, or `robot_1`.
- `--mask-overlay-alpha`: green overlay alpha.
- `--mask-overlay-output-mode`: `overlay` replaces `image[view]`; `mask_image` keeps RGB unchanged and adds `image[f"{view}_mask"]`.
- `--mask-overlay-mask-key-suffix`: suffix for mask-image mode, default `_mask`.
- `--mask-overlay-tracking-mode`: `image` for previous bbox/logits image prompting, `video_window` for rolling SAM3 video tracking, or `text_select_video` for text-detect plus clicked-instance selection.
- `--mask-overlay-video-window-size`: number of recent frames to rebuild as a video session in `video_window` mode.
- `--sam3-video-version`: SAM3 video predictor version, usually `sam3` for `sam3.pt` checkpoints.
- `--sam3-checkpoint-path`: optional local SAM3 checkpoint path.
- `--sam3-device`: SAM3 device, default `cuda`.
- `--sam3-path`: optional SAM3 source path; defaults to `third_party/sam3`.

Startup will fail fast if the local checkpoint path does not exist or if SAM3 cannot be imported. If `--sam3-checkpoint-path` is omitted, SAM3 may download its default checkpoint during server startup.

`video_window` mode loads both the SAM3 image interactivity model and the SAM3 video predictor. The image model is used only for the initial clicked-point segmentation; the resulting bbox becomes the visual prompt for the video tracker. This is more stable than repeatedly prompting the image model with a fixed point or previous bbox, but it is also slower and uses more GPU memory.

`text_select_video` mode uses the SAM3 video predictor's text prompt on the first frame of each short window, selects the target instance using either the initial clicked point or the previous tracked bbox, propagates all text-matched instances, and keeps the candidate with the strongest bbox continuity. It requires the rollout client to pass `--mask_prompt_text`.

## Start Rollout With Click Prompt

Example:

```bash
python scripts/pi0_rollout_client_fasttouch_rpy.py \
  --description "..." \
  --arm_mode dual \
  --mask_overlay \
  --mask_view robot_0 \
  --mask_prompt_text "sponge"
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

For `text_select_video`, always pass a text prompt:

```bash
--mask_prompt_text "sponge"
```

If the object moves too much between two policy calls, enable dense SAM-only tracking frames from the rollout client:

```bash
--mask_track_between_actions \
--mask_track_every_n_actions 1
```

With this enabled, the client sends the selected camera's 224x224 rollout image to the server after every N executed actions. These requests use `track_only=True`, so the server updates SAM3 tracking state but does not run policy inference. This gives SAM3 a denser video stream while keeping the policy call rate unchanged.

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

During rollout, every policy inference also saves `rollout/<infer_index>_<view>_server.png`, `rollout/<infer_index>_<view>_overlay.png`, and `rollout/<infer_index>_<view>_mask.png`. Dense `track_only` debug frames use indices starting at `1000000` to avoid overwriting policy inference frames.

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
    "text": "sponge",
    "preview_only": True,
    "alpha": 0.35,
}
```

Preview response:

```python
resp["__mask_overlay"] = {
    "view": "robot_0",
    "tracking_mode": "text_select_video",
    "mask": mask_uint8,
    "overlay": overlay_rgb_uint8,
    "score": score_or_none,
    "bbox_xyxy": bbox_or_none,
    "initialized": True,
    "needs_reprompt": False,
    "reprompt_reason": None,
    "selected_obj_id": selected_obj_id_or_none,
    "candidate_count": candidate_count_or_none,
}
```

For normal rollout after confirmation, the client sends:

```python
obs["__mask_overlay"] = {
    "enabled": True,
    "view": "robot_0",
    "text": "sponge",
    "alpha": 0.35,
}
```

Dense tracking-only request sent between policy calls:

```python
obs["__mask_overlay"] = {
    "enabled": True,
    "view": "robot_0",
    "text": "sponge",
    "track_only": True,
    "alpha": 0.35,
}
```

With `--mask-overlay-output-mode overlay`, the server replaces `obs["image"][view]` with the overlay image before calling the policy. With `--mask-overlay-output-mode mask_image`, the server keeps the RGB image unchanged and adds a per-view mask image key:

```python
obs["image"]["front"]      # RGB camera image
obs["image"]["front_mask"] # SAM3 binary mask as RGB image
```

The `sponge_visual_mask_keys_*` training configs convert dataset masks into six model-side image slots:

```python
base_0_rgb
left_wrist_0_rgb
right_wrist_0_rgb
base_0_mask_rgb
left_wrist_0_mask_rgb
right_wrist_0_mask_rgb
```

Missing RGB or mask views are present as zero images with `image_mask=False`, so they do not contribute tokens to the model. The server attaches lightweight mask metadata to the policy response and only returns full `mask` / `overlay` arrays for preview or when the client explicitly requests debug images. If `text_select_video` loses the selected instance, the server returns only `__mask_overlay` with `needs_reprompt=True`; the rollout client pauses policy execution, opens the click window on the current frame, resets the tracker, and then continues.

## Implementation Notes

- The SAM3 wrapper lives in `src/openpi/serving/mask_overlay.py`.
- The server integration lives in `scripts/serve_policy.py`.
- The rollout click/preview flow lives in `scripts/pi0_rollout_client_fasttouch_rpy.py` and `scripts/pi0_rollout_client_xarm_rpy.py`.
- `image` mode uses SAM3 image interactivity with the previous mask/logits/bounding box as the next prompt.
- `video_window` mode uses SAM3's video predictor, but SAM3 initializes video state from a fixed image list/video path rather than an appendable stream. The server therefore rebuilds a short session from the latest frames for each policy inference, prompts frame 0 with the tracked bbox, and propagates forward to the newest frame.
- `text_select_video` mode uses text to detect instances in frame 0 of each short session, picks one instance by clicked point or previous bbox, propagates all text-matched objects, and chooses the best continuity match on each propagated frame. Point and text are not sent to SAM3 as one combined point prompt; point is used for instance selection after text detection.
- `track_only` requests update the SAM3 state without calling the wrapped OpenPI policy. Use them to feed denser frames during one rollout action chunk. The tracking frames should stay at the same resolution as policy images; the current clients already downsample camera frames to 224x224 before sending.

## Troubleshooting

- If the client raises `server 未启用 --mask-overlay`, restart `serve_policy.py` with `--mask-overlay`.
- If SAM3 cannot download weights, authenticate Hugging Face or pass `--sam3-checkpoint-path`.
- If segmentation drifts in `image` mode, try `--mask-overlay-tracking-mode text_select_video --mask-overlay-video-window-size 8` on the server and `--mask_prompt_text "sponge"` on the client.
- If `text_select_video` returns `needs_reprompt=True`, the client will ask for a new click on the current frame. If this happens often, make the text prompt more specific or lower the camera motion between policy calls.
- If `video_window` fails at startup, check that `--sam3-video-version` matches the checkpoint family. For `facebook/sam3` / `sam3.pt`, use `--sam3-video-version sam3`.
- If GPU memory is tight, run SAM3 and OpenPI on a larger GPU or use a separate server process/GPU for overlay experiments.
