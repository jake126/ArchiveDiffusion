\# Command log



This file records command-line steps used to construct the ArchiveDiffusion pilot dataset. Raw videos, extracted frames, selected frames, and processed datasets are excluded from Git. Commands are logged here so the data pipeline can be reproduced.



\## 1. Video download



```powershell

Invoke-WebRequest `                

>>   -Uri "https://archive.org/download/Nosferatu\_DVD\_quality/nosferatu-1of5\_512kb.mp4" `

>>   -OutFile ".\\data\\raw\_videos\\nosferatu\_1of5\_512kb.mp4"    ```



\## 2. Frame extraction



```powershell

ffmpeg `                           

>>   -ss 00:00:00 `                                                                      

>>   -t 00:20:00 `                                                   

>>   -i ".\\data\\raw\_videos\\nosferatu\_1of5\_512kb.mp4" `                                   

>>   -vf "fps=1/2,scale=512:-1,format=gray" `                                          

>>   ".\\data\\raw\_frames\\nosferatu\_1of5\_20min\_fps0\_5\\nosferatu\_%06d.png"

```



\## 3. Environment setup



```Anaconda Prompt due to EasyOCR library clashes in VS code

```

conda create -n archivediffusion python=3.10 -y
conda activate archivediffusion
python -c "import ssl; print(ssl.OPENSSL_VERSION)"

python -m pip install --upgrade pip setuptools wheel
python -m pip install easyocr opencv-python pillow numpy tqdm

python ".\scripts\filter_text_frames_easyocr.py" ^
  --input_dir ".\data\raw_frames\nosferatu_1of5_20min_fps0_5" ^
  --output_csv ".\evaluations\results\nosferatu_easyocr_text_filter.csv" ^
  --review_sheet ".\evaluations\figures\nosferatu_easyocr_flagged_text_frames.png" ^
  --keep_dir ".\data\selected_frames\nosferatu_non_text_easyocr" ^
  --text_dir ".\data\selected_frames\nosferatu_text_easyocr" ^
  --min_confidence 0.35


\## 4. OCR filtering



```Anaconda Prompt

python ".\\scripts\\filter\_text\_frames.py" ^

&#x20; --input\_dir ".\\data\\raw\_frames\\nosferatu\_1of5\_20min\_fps0\_5" ^

&#x20; --output\_csv ".\\evaluations\\results\\nosferatu\_easyocr\_text\_filter.csv" ^

&#x20; --review\_sheet ".\\evaluations\\figures\\nosferatu\_easyocr\_flagged\_text\_frames.png" ^

&#x20; --keep\_dir ".\\data\\selected\_frames\\nosferatu\_non\_text\_easyocr" ^

&#x20; --text\_dir ".\\data\\selected\_frames\\nosferatu\_text\_easyocr" ^

&#x20; --min\_confidence 0.35

```



\## 5. Grain/degradation ranking



```powershell

".\\scripts\\rank\_grainy\_frames.py" `                                                                                  

>>   --input\_dir ".\\data\\selected\_frames\\nosferatu\_non\_title\_cards" `

>>   --output\_csv ".\\evaluations\\results\\nosferatu\_non\_title\_grain\_rankings.csv" `       

>>   --output\_sheet ".\\evaluations\\figures\\nosferatu\_non\_title\_top\_grainy\_frames.png" `

>>   --top\_k 100                

```



\## 6. Candidate set creation



```powershell

New-Item -ItemType Directory -Force ".\data\curated\nosferatu_cleanish_candidates"
New-Item -ItemType Directory -Force ".\data\curated\nosferatu_degraded_candidates"
New-Item -ItemType Directory -Force ".\evaluations\results"
New-Item -ItemType Directory -Force ".\evaluations\figures"

python ".\scripts\create_degradation_candidate_sets.py" `
  --input_dir ".\data\selected_frames\nosferatu_non_text_easyocr" `
  --output_csv ".\evaluations\results\nosferatu_degradation_ranking.csv" `
  --cleanish_dir ".\data\curated\nosferatu_cleanish_candidates" `
  --degraded_dir ".\data\curated\nosferatu_degraded_candidates" `
  --cleanish_sheet ".\evaluations\figures\nosferatu_cleanish_candidates_contact_sheet.png" `
  --degraded_sheet ".\evaluations\figures\nosferatu_degraded_candidates_contact_sheet.png" `
  --n_cleanish 100 `
  --n_degraded 100

```



\## 7. Contact-sheet generation



```powershell

".\\scripts\\make\_contact\_sheet.py" `                                                                                  

>>   --input\_dir ".\\data\\raw\_frames\\nosferatu\_1of5\_20min\_fps0\_5" `   

>>   --output\_path ".\\outputs\\contact\_sheets\\nosferatu\_20min\_overview.png" `             

>>   --every 10 `                                                                      

>>   --max\_images 100   

```

\## 8. Artificial blemish augmentation

```powershell
New-Item -ItemType Directory -Force ".\data\processed\synthetic_pairs\nosferatu_cleanish"

python ".\scripts\create_synthetic_blemish_examples.py" `
  --cleanish_dir ".\data\curated\nosferatu_cleanish_candidates" `
  --output_dir ".\data\processesd\synthetic_pairs\nosferatu_cleanish" `
  --example_index 1 `
  --seed 42
```

\## 9. Creating three image use-case example
New-Item -ItemType Directory -Force ".\outputs\figures"

python ".\scripts\create_three_use_case_visual.py" `
  --synthetic_dir ".\data\processed\synthetic_pairs\nosferatu_cleanish" `
  --natural_degraded_dir ".\data\curated\nosferatu_degraded_candidates" `
  --output_path ".\outputs\figures\archive_diffusion_three_use_cases.png"
```powershell

\## 10. synthetic datasets (01/06/2026)
python ".\scripts\create_synthetic_restoration_dataset.py" `
  --cleanish_dir ".\data\curated\nosferatu_cleanish_candidates" `
  --output_dir ".\data\processed\synthetic_restoration\nosferatu_v0" `
  --n_images 50 `
  --image_size 128 `
  --seed 42

\## 11. baseline restoration for benchmarking (01/06/2026)
python ".\scripts\run_baseline_restoration.py" `
  --dataset_dir ".\data\processed\synthetic_restoration\nosferatu_v0" `
  --output_dir ".\outputs\predictions\baselines_nosferatu_v0"

\## 12. evaluate baseline restorations (01/06/2026)
New-Item -ItemType Directory -Force ".\evaluations\results"
New-Item -ItemType Directory -Force ".\outputs\evaluation_grids"

python ".\scripts\evaluate_restoration_outputs.py" `
  --prediction_manifest ".\outputs\predictions\baselines_nosferatu_v0\prediction_manifest.csv" `
  --output_metrics ".\evaluations\results\baseline_restoration_metrics.csv" `
  --output_summary ".\evaluations\results\baseline_restoration_summary.csv" `
  --output_grid ".\outputs\evaluation_grids\baseline_restoration_examples.png"