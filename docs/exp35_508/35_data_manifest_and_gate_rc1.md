# 35_data_manifest_and_gate_rc1

- generated_at: 2026-10-26T23:56:02
- status: read-only manifest + count-only; NOT frozen
- data_root: /home/molly/projects/danmaku
- official_candidate: ODS + dmid dedup; DWD = sensitivity only
- no training, no D-conf, no data modification

## 1. Batches

| batch | ODS rows | ODS videos |
|---|---:|---:|
| 20260902 | 129976 | 91 |
| 20260907 | 181018 | 118 |
| 20260914 | 249685 | 200 |
| 20260917 | 116252 | 100 |

File inventory and SHA-256: file_manifest.csv / manifest.sha256

## 2. dmid audit

- ODS rows: 676931
- ODS videos: 508
- dmid null: 0
- dmid length 19: 676931
- dmid unique: 674498
- duplicate rows: 4866 ; duplicate IDs: 2433
- same dmid different bvid groups: 0
- duplicate source: {'BV1mwY96jEom': 4866}
- detail: dmid_cross_batch_duplicates.csv

## 3. 11 old-batch-only rows

- BV1mwY96jEom: 0914-only rows not in 0917; pool uses latest, so excluded from pool.
- detail: old_batch_only_11_rows.csv

## 4. count-only

| metric | ODS+dmid | DWD |
|---|---:|---:|
| videos | 409 | 409 |
| points | 13049 | 13049 |
| pos | 4667 | 4656 |
| rate | 0.3577 | 0.3568 |
| both | 371 | 370 |
| ge5 | 318 | 317 |
| ge10 | 246 | 245 |

| split | ODS videos | ODS points | ODS pos | ODS ge5 both |
|---|---:|---:|---:|---:|
| train | 175 | 5460 | 2030 | 139 |
| dev | 234 | 7589 | 2637 | 179 |

## 5. Candidate split

- D-train-508: 20260902, 20260907
- D-dev-508: 20260914, 20260917
- D_train_end: 2026-09-07
- train videos: 175 ; dev videos: 234 ; intersection: 0
- lists: D-train-508_bvids.txt / D-dev-508_bvids.txt

## 6. Gate candidate

- per-video >=5 valid decision points and both classes
- per-video min positives/negatives, min evaluable videos, D-conf min events: pending approval
- D-conf >=150 eligible dedup single-P videos
- primary: video-macro within-video AUC; bvid cluster bootstrap; calibration independent gate

## 7. Data semantics

- ODS 676,931; DWD 672,439; pool 669,995
- ODS to DWD: 4,492 rows removed by (bvid,text,time_offset) dedup; may include real repeats; DWD sensitivity only
- DWD to pool: 2,444 rows = latest-per-bvid pool; BV1mwY96jEom has 2,433 overlap + 11 old-only rows

## 8. Trace

- generator: read-only Python on VM
- git HEAD: 8c3d1ec (observed; working tree not clean)
- no training / no D-conf