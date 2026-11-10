# Who Moves? – detecting objects that go somewhere in surveillance video

My solution to the **MALTO Recruitment Challenge: "Who Moves?"** on Kaggle (October 2026).
**10th place on the private leaderboard.** Public F-score: baseline 0.603 → ours 0.845.

The task: in parking-lot camera videos, raise one alarm for every person, vehicle or animal that **goes somewhere**, and stay silent for everything else: parked cars, shadows, snow, camera shake, swaying branches, people moving in place. False alarms on known non-targets are penalised twice.

## How it works

```
video (10 fps)
  → 3 detector passes per frame, merged by same-class NMS
      1. COCO YOLO11x, full frame (1280 px)
      2. COCO YOLO11x, 2×2 overlapping tiles (small objects ~1.4× larger)
      3. fine-tuned YOLO11x, person boxes only (confidence ≥ 0.25)
  → own tracker: Kalman filter + Hungarian matching on every frame
      confidence ≥ 0.4 starts a track, 0.1–0.4 only extends one
  → rule: alarm once a track is ≥ 1 s old and has moved more than
          its own box height from its start (free box edge for border boxes)
  → one alarm per track, using only past frames
```

## Key ideas

- **Scene-based validation.** 81 of 90 training videos come from two parking lots, and the test scenes are unseen. A random split leaks backgrounds, so every choice was checked with a scene split (lot A / lot B / other). For example, a learned motion classifier beat the rule on a random split (0.937 vs 0.906), the gain almost vanished on the scene split (0.911), and it lost on public (0.797 vs 0.804).
- **Error analysis first.** About two thirds of the loss came from small people (20–50 px) the detector never saw. Feeding the pipeline the labelled boxes of people under 40 px raised F from 0.835 to 0.970, so the bottleneck was the detector, not the tracker or the rule.
- **Fine-tuning on held-out scenes.** YOLO11x was fine-tuned on lot A and the other scenes; lot B was never used for training. On lot B, F rose from 0.890 to 0.921 (misses 63 → 39, false alarms 19 → 28).

**Public score progression:** baseline 0.603 → YOLO11x 0.703 → own tracker + rule 0.804 → 2×2 tiles 0.814 → fine-tuned person detector 0.845.

## Files

| File | What it is |
|---|---|
| `who_moves_final_pipeline.ipynb` | Final submission (0.845 public). Runs end to end on Kaggle: fine-tuning, detection, tracking, decision. ~5 h on 2× T4. |
| `report.ipynb` | The competition report: what "moving" means, what must never alarm, validation, key choices with concrete cases (video id + second), error analysis, and the ideas that did not work. |

## Running it

The notebook is meant for Kaggle: attach the competition data, turn on GPU (T4 ×2) and Internet (the COCO weights `yolo11x.pt` are downloaded from Ultralytics), and run all cells.

**Data is not included.** The videos and labels belong to the competition and are not redistributed here.

## Models

- YOLO11x (Ultralytics, COCO-pretrained `yolo11x.pt`, AGPL-3.0), fine-tuned on frames of the competition training videos (one frame every 2 s, 23 epochs, first 10 layers frozen, AdamW lr 0.001, image size 1280, fixed seed, deterministic mode; the best epoch on lot B is kept).
- Also tried and dropped: Faster R-CNN, RT-DETR R50, YOLO11m, a HistGradientBoosting motion classifier.
- No VLM, CLIP, open-vocabulary or video models. Test videos were never watched or labelled.

## License

AGPL-3.0, because the project builds on Ultralytics YOLO (AGPL-3.0).
