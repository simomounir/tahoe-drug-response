# Phase 1 QC report

Conditions retained: 2123 / 2213
Dropped conditions: 90 (see table below)
Controls retained: 24 / 24
Mean library size: 5,929,529
Status: pass
Treated conditions with logFC vs plate-matched DMSO: 2099
Output size: 418.38 MB
Zero-count genes across the slice: 9777
Zero-count gene sample: 5S_rRNA, 5S_rRNA-2, 5S_rRNA-3, 5S_rRNA-4, 5S_rRNA-5, 5S_rRNA-6, 5S_rRNA-7, 5S_rRNA-8, 5S_rRNA-9, 5_8S_rRNA, 5_8S_rRNA-1, 5_8S_rRNA-4, 5_8S_rRNA-8, 7SK-2, 7SK-3, 7SK-4, 7SK-5, AA06, ABHD17AP9, ABITRAMP1

## Dropped conditions

| cell_line | plate | drug_name | dose | n_cells | n_cells_atlas | reason |
|---|---|---|---|---|---|---|
| CVCL_0293 | plate1 | Abemaciclib | 0.05 | 2 | 2 | n_cells 2 < 100 |
| CVCL_0293 | plate1 | Belzutifan | 0.05 | 54 | 54 | n_cells 54 < 100 |
| CVCL_0293 | plate1 | DTP3 | 0.05 | 5 | 5 | n_cells 5 < 100 |
| CVCL_0293 | plate2 | Encorafenib | 0.5 | 6 | 6 | n_cells 6 < 100 |
| CVCL_0293 | plate2 | Erdafitinib  | 0.5 | 11 | 11 | n_cells 11 < 100 |
| CVCL_0293 | plate1 | Hydroxyfasudil | 0.05 | 26 | 26 | n_cells 26 < 100 |
| CVCL_0293 | plate3 | Lonafarnib | 5.0 | 93 | 93 | n_cells 93 < 100 |
| CVCL_0293 | plate1 | PF-06260933 | 0.05 | 7 | 7 | n_cells 7 < 100 |
| CVCL_0293 | plate2 | Pemigatinib | 0.5 | 95 | 95 | n_cells 95 < 100 |
| CVCL_0293 | plate2 | SBI-0640756 | 0.5 | 25 | 25 | n_cells 25 < 100 |
| CVCL_0293 | plate1 | Sonidegib | 0.05 | 29 | 29 | n_cells 29 < 100 |
| CVCL_0293 | plate1 | c-Kit-IN-1 | 0.05 | 3 | 3 | n_cells 3 < 100 |
| CVCL_0293 | plate1 | olaparib | 0.05 | 4 | 4 | n_cells 4 < 100 |
| CVCL_0371 | plate1 | Belzutifan | 0.05 | 48 | 48 | n_cells 48 < 100 |
| CVCL_0371 | plate1 | DTP3 | 0.05 | 4 | 4 | n_cells 4 < 100 |
| CVCL_0371 | plate2 | Encorafenib | 0.5 | 7 | 7 | n_cells 7 < 100 |
| CVCL_0371 | plate2 | Erdafitinib  | 0.5 | 2 | 2 | n_cells 2 < 100 |
| CVCL_0371 | plate1 | Hydroxyfasudil | 0.05 | 22 | 22 | n_cells 22 < 100 |
| CVCL_0371 | plate3 | Lonafarnib | 5.0 | 12 | 12 | n_cells 12 < 100 |
| CVCL_0371 | plate1 | Methylprednisolone succinate | 0.05 | 1 | 1 | n_cells 1 < 100 |
| CVCL_0371 | plate2 | Pemigatinib | 0.5 | 65 | 65 | n_cells 65 < 100 |
| CVCL_0371 | plate2 | SBI-0640756 | 0.5 | 14 | 14 | n_cells 14 < 100 |
| CVCL_0371 | plate1 | Sonidegib | 0.05 | 14 | 14 | n_cells 14 < 100 |
| CVCL_0371 | plate1 | olaparib | 0.05 | 1 | 1 | n_cells 1 < 100 |
| CVCL_0399 | plate1 | Abemaciclib | 0.05 | 2 | 2 | n_cells 2 < 100 |
| CVCL_0399 | plate1 | DTP3 | 0.05 | 3 | 3 | n_cells 3 < 100 |
| CVCL_0399 | plate2 | Encorafenib | 0.5 | 18 | 18 | n_cells 18 < 100 |
| CVCL_0399 | plate2 | Erdafitinib  | 0.5 | 8 | 8 | n_cells 8 < 100 |
| CVCL_0399 | plate1 | Hydroxyfasudil | 0.05 | 46 | 46 | n_cells 46 < 100 |
| CVCL_0399 | plate1 | Methylprednisolone succinate | 0.05 | 2 | 2 | n_cells 2 < 100 |
| CVCL_0399 | plate1 | PF-06260933 | 0.05 | 4 | 4 | n_cells 4 < 100 |
| CVCL_0399 | plate1 | Ralimetinib dimesylate | 0.05 | 2 | 2 | n_cells 2 < 100 |
| CVCL_0399 | plate2 | SBI-0640756 | 0.5 | 31 | 31 | n_cells 31 < 100 |
| CVCL_0399 | plate1 | Sonidegib | 0.05 | 38 | 38 | n_cells 38 < 100 |
| CVCL_0399 | plate1 | olaparib | 0.05 | 9 | 9 | n_cells 9 < 100 |
| CVCL_0459 | plate1 | Abemaciclib | 0.05 | 12 | 12 | n_cells 12 < 100 |
| CVCL_0459 | plate1 | Belzutifan | 0.05 | 80 | 80 | n_cells 80 < 100 |
| CVCL_0459 | plate1 | DTP3 | 0.05 | 8 | 8 | n_cells 8 < 100 |
| CVCL_0459 | plate2 | Encorafenib | 0.5 | 10 | 10 | n_cells 10 < 100 |
| CVCL_0459 | plate2 | Erdafitinib  | 0.5 | 37 | 37 | n_cells 37 < 100 |
| CVCL_0459 | plate1 | Hydroxyfasudil | 0.05 | 81 | 81 | n_cells 81 < 100 |
| CVCL_0459 | plate1 | Methylprednisolone succinate | 0.05 | 4 | 4 | n_cells 4 < 100 |
| CVCL_0459 | plate1 | PF-06260933 | 0.05 | 28 | 28 | n_cells 28 < 100 |
| CVCL_0459 | plate1 | Ralimetinib dimesylate | 0.05 | 13 | 13 | n_cells 13 < 100 |
| CVCL_0459 | plate2 | SBI-0640756 | 0.5 | 64 | 64 | n_cells 64 < 100 |
| CVCL_0459 | plate1 | Sonidegib | 0.05 | 47 | 47 | n_cells 47 < 100 |
| CVCL_0459 | plate1 | c-Kit-IN-1 | 0.05 | 14 | 14 | n_cells 14 < 100 |
| CVCL_0459 | plate1 | olaparib | 0.05 | 14 | 14 | n_cells 14 < 100 |
| CVCL_0480 | plate1 | Belzutifan | 0.05 | 18 | 18 | n_cells 18 < 100 |
| CVCL_0480 | plate2 | Erdafitinib  | 0.5 | 3 | 3 | n_cells 3 < 100 |
| CVCL_0480 | plate1 | Hydroxyfasudil | 0.05 | 15 | 15 | n_cells 15 < 100 |
| CVCL_0480 | plate1 | PF-06260933 | 0.05 | 1 | 1 | n_cells 1 < 100 |
| CVCL_0480 | plate2 | Pemigatinib | 0.5 | 44 | 44 | n_cells 44 < 100 |
| CVCL_0480 | plate2 | SBI-0640756 | 0.5 | 13 | 13 | n_cells 13 < 100 |
| CVCL_0480 | plate1 | Sonidegib | 0.05 | 24 | 24 | n_cells 24 < 100 |
| CVCL_0480 | plate1 | olaparib | 0.05 | 1 | 1 | n_cells 1 < 100 |
| CVCL_0546 | plate1 | Abemaciclib | 0.05 | 3 | 3 | n_cells 3 < 100 |
| CVCL_0546 | plate1 | DTP3 | 0.05 | 6 | 6 | n_cells 6 < 100 |
| CVCL_0546 | plate2 | Encorafenib | 0.5 | 23 | 23 | n_cells 23 < 100 |
| CVCL_0546 | plate2 | Erdafitinib  | 0.5 | 24 | 24 | n_cells 24 < 100 |
| CVCL_0546 | plate1 | Hydroxyfasudil | 0.05 | 54 | 54 | n_cells 54 < 100 |
| CVCL_0546 | plate1 | PF-06260933 | 0.05 | 7 | 7 | n_cells 7 < 100 |
| CVCL_0546 | plate1 | Ralimetinib dimesylate | 0.05 | 5 | 5 | n_cells 5 < 100 |
| CVCL_0546 | plate2 | SBI-0640756 | 0.5 | 40 | 40 | n_cells 40 < 100 |
| CVCL_0546 | plate1 | Sonidegib | 0.05 | 47 | 47 | n_cells 47 < 100 |
| CVCL_0546 | plate1 | c-Kit-IN-1 | 0.05 | 3 | 3 | n_cells 3 < 100 |
| CVCL_0546 | plate1 | olaparib | 0.05 | 5 | 5 | n_cells 5 < 100 |
| CVCL_1056 | plate1 | Belzutifan | 0.05 | 7 | 7 | n_cells 7 < 100 |
| CVCL_1056 | plate2 | Encorafenib | 0.5 | 2 | 2 | n_cells 2 < 100 |
| CVCL_1056 | plate2 | Erdafitinib  | 0.5 | 1 | 1 | n_cells 1 < 100 |
| CVCL_1056 | plate1 | Hydroxyfasudil | 0.05 | 14 | 14 | n_cells 14 < 100 |
| CVCL_1056 | plate1 | PF-06260933 | 0.05 | 1 | 1 | n_cells 1 < 100 |
| CVCL_1056 | plate2 | Pemigatinib | 0.5 | 26 | 26 | n_cells 26 < 100 |
| CVCL_1056 | plate2 | SBI-0640756 | 0.5 | 9 | 9 | n_cells 9 < 100 |
| CVCL_1056 | plate1 | Sonidegib | 0.05 | 10 | 10 | n_cells 10 < 100 |
| CVCL_1056 | plate1 | c-Kit-IN-1 | 0.05 | 1 | 1 | n_cells 1 < 100 |
| CVCL_1056 | plate1 | olaparib | 0.05 | 1 | 1 | n_cells 1 < 100 |
| CVCL_1285 | plate1 | Abemaciclib | 0.05 | 2 | 2 | n_cells 2 < 100 |
| CVCL_1285 | plate1 | Belzutifan | 0.05 | 17 | 17 | n_cells 17 < 100 |
| CVCL_1285 | plate2 | Encorafenib | 0.5 | 1 | 1 | n_cells 1 < 100 |
| CVCL_1285 | plate2 | Erdafitinib  | 0.5 | 3 | 3 | n_cells 3 < 100 |
| CVCL_1285 | plate1 | Hydroxyfasudil | 0.05 | 15 | 15 | n_cells 15 < 100 |
| CVCL_1285 | plate1 | Methylprednisolone succinate | 0.05 | 1 | 1 | n_cells 1 < 100 |
| CVCL_1285 | plate1 | PF-06260933 | 0.05 | 3 | 3 | n_cells 3 < 100 |
| CVCL_1285 | plate2 | Pemigatinib | 0.5 | 59 | 59 | n_cells 59 < 100 |
| CVCL_1285 | plate1 | Ralimetinib dimesylate | 0.05 | 1 | 1 | n_cells 1 < 100 |
| CVCL_1285 | plate2 | SBI-0640756 | 0.5 | 25 | 25 | n_cells 25 < 100 |
| CVCL_1285 | plate1 | Sonidegib | 0.05 | 10 | 10 | n_cells 10 < 100 |
| CVCL_1285 | plate1 | c-Kit-IN-1 | 0.05 | 8 | 8 | n_cells 8 < 100 |
| CVCL_1285 | plate1 | olaparib | 0.05 | 2 | 2 | n_cells 2 < 100 |

## Per cell line

| cell_line | conditions | kept | median_cells | median_library_size |
|---|---|---|---|---|
| CVCL_0293 | 277 | 264 | 2,069 | 5,220,754 |
| CVCL_0371 | 275 | 264 | 1,589 | 3,404,702 |
| CVCL_0399 | 278 | 267 | 2,336 | 4,999,964 |
| CVCL_0459 | 279 | 266 | 3,299 | 5,591,754 |
| CVCL_0480 | 273 | 265 | 1,834 | 5,446,725 |
| CVCL_0546 | 278 | 267 | 3,496 | 6,464,406 |
| CVCL_1056 | 275 | 265 | 1,240 | 4,133,768 |
| CVCL_1285 | 278 | 265 | 1,674 | 3,428,601 |

## Controls

| cell_line | plate | dose | n_cells | n_cells_atlas | library_size |
|---|---|---|---|---|---|
| CVCL_0293 | plate1 | 0.0 | 5865 | 5865 | 16575201 |
| CVCL_0293 | plate2 | 0.0 | 8055 | 8055 | 26085162 |
| CVCL_0293 | plate3 | 0.0 | 2448 | 2448 | 5942477 |
| CVCL_0371 | plate1 | 0.0 | 4864 | 4864 | 10571870 |
| CVCL_0371 | plate2 | 0.0 | 5886 | 5886 | 16319730 |
| CVCL_0371 | plate3 | 0.0 | 2057 | 2057 | 3251957 |
| CVCL_0399 | plate1 | 0.0 | 7271 | 7271 | 16752406 |
| CVCL_0399 | plate2 | 0.0 | 9330 | 9330 | 25797374 |
| CVCL_0399 | plate3 | 0.0 | 2592 | 2592 | 5464866 |
| CVCL_0459 | plate1 | 0.0 | 9469 | 9469 | 16407712 |
| CVCL_0459 | plate2 | 0.0 | 14583 | 14583 | 32489330 |
| CVCL_0459 | plate3 | 0.0 | 2222 | 2222 | 2635787 |
| CVCL_0480 | plate1 | 0.0 | 5341 | 5341 | 16902159 |
| CVCL_0480 | plate2 | 0.0 | 8795 | 8795 | 34469170 |
| CVCL_0480 | plate3 | 0.0 | 3386 | 3386 | 6263469 |
| CVCL_0546 | plate1 | 0.0 | 10609 | 10609 | 19677151 |
| CVCL_0546 | plate2 | 0.0 | 14701 | 14701 | 33824399 |
| CVCL_0546 | plate3 | 0.0 | 3059 | 3059 | 4455101 |
| CVCL_1056 | plate1 | 0.0 | 3761 | 3761 | 12963629 |
| CVCL_1056 | plate2 | 0.0 | 6793 | 6793 | 30256389 |
| CVCL_1056 | plate3 | 0.0 | 2773 | 2773 | 5343294 |
| CVCL_1285 | plate1 | 0.0 | 4707 | 4707 | 9813878 |
| CVCL_1285 | plate2 | 0.0 | 8616 | 8616 | 23039245 |
| CVCL_1285 | plate3 | 0.0 | 2489 | 2489 | 3371824 |