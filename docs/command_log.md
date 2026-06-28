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

\## 13. Conditional DDPM training

```powershell
python ".\scripts\train_conditional_ddpm.py" `
  --dataset_dir ".\data\processed\synthetic_restoration\nosferatu_v0" `
  --output_dir ".\outputs\models\conditional_ddpm_nosferatu_v0" `
  --image_size 128 `
  --batch_size 8 `
  --epochs 50 `
  --learning_rate 1e-4 `
  --seed 42

\## 14. Conditional DDPM Sampling
python ".\scripts\sample_conditional_ddpm.py" `
  --model_dir ".\outputs\models\conditional_ddpm_nosferatu_v0_cpu_test\final" `
  --dataset_dir ".\data\processed\synthetic_restoration\nosferatu_v0" `
  --output_dir ".\outputs\predictions\conditional_ddpm_nosferatu_v0" `
  --method_name "conditional_ddpm_50_steps" `
  --num_inference_steps 50

\## 15. Conditional DDPM Evaluation
python ".\scripts\evaluate_restoration_outputs.py" `
  --prediction_manifest ".\outputs\predictions\conditional_ddpm_nosferatu_v0\prediction_manifest.csv" `
  --output_metrics ".\evaluations\results\conditional_ddpm_restoration_metrics.csv" `
  --output_summary ".\evaluations\results\conditional_ddpm_restoration_summary.csv" `
  --output_grid ".\outputs\evaluation_grids\conditional_ddpm_restoration_examples.png"

\## 16. Train test split
python ".\scripts\create_train_val_test_split.py" `
   --metadata_path ".\data\processed\synthetic_restoration\nosferatu_v0\metadata.csv" `
   --output_path ".\data\processed\synthetic_restoration\nosferatu_v0\metadata_with_splits.csv" `
   --train_fraction 0.70 `
   --val_fraction 0.15 `
   --test_fraction 0.15 `
   --seed 42

\## 17. Training loss curves
python ".\scripts\plot_training_curves.py" `
  --run_dir ".\outputs\models\conditional_ddpm_nosferatu_v0_cpu_test" `
  --output_path ".\outputs\training_curves\conditional_ddpm_v0_cpu_test_loss.png" `
  --title "Conditional DDPM v0 CPU test training loss"

  python ".\scripts\plot_training_curves.py" `
  --run_dir ".\outputs\models\conditional_ddpm_nosferatu_v1_cpu_long" `
  --output_path ".\outputs\training_curves\conditional_ddpm_v1_cpu_long_loss.png" `
  --title "Conditional DDPM v1 CPU long-run training loss"

\## 18. Training a split-aware model
python ".\scripts\train_conditional_ddpm.py" `
  --dataset_dir ".\data\processed\synthetic_restoration\nosferatu_v0" `
  --metadata_file "metadata_with_splits.csv" `
  --train_split train `
  --val_split val `
  --output_dir ".\outputs\models\conditional_ddpm_nosferatu_v2_splitaware" `
  --image_size 128 `
  --batch_size 2 `
  --epochs 150 `
  --learning_rate 1e-4 `
  --num_train_timesteps 1000 `
  --seed 42 `
  --save_every_epochs 50 `
  --use_mlflow `
  --mlflow_tracking_uri "sqlite:///mlflow.db" `
  --mlflow_experiment_name "ArchiveDiffusion" `
  --run_name "conditional_ddpm_v2_splitaware_150epochs"

\## 19. Sampling predictions of the retrained train/test split model
python ".\scripts\sample_conditional_ddpm.py" `
  --model_dir ".\outputs\models\conditional_ddpm_nosferatu_v2_splitaware\final" `
  --dataset_dir ".\data\processed\synthetic_restoration\nosferatu_v0" `
  --metadata_file ".\metadata_with_splits.csv" `
  --split test `
  --output_dir ".\outputs\predictions\conditional_ddpm_nosferatu_v2_test_50steps" `
  --method_name "conditional_ddpm_v2_test_50_steps" `
  --num_inference_steps 50

\## 20. Generate blinded review widget
python ".\scripts\create_human_review_sheet.py" `
  --metadata_file ".\data\processed\synthetic_restoration\nosferatu_v0\metadata_with_splits.csv" `
  --split test `
  --prediction_manifest ".\outputs\predictions\baselines_nosferatu_v0\prediction_manifest.csv" `
  --prediction_manifest ".\outputs\predictions\conditional_ddpm_nosferatu_v1_50steps\prediction_manifest.csv" `
  --prediction_manifest ".\outputs\predictions\conditional_ddpm_nosferatu_v1_100steps\prediction_manifest.csv" `
  --prediction_manifest ".\outputs\predictions\conditional_ddpm_nosferatu_v2_test_50steps\prediction_manifest.csv" `
  --output_dir ".\evaluations\human_review\review_sheets\ddpm_v2_blinded" `
  --output_csv ".\evaluations\human_review\ddpm_v2_review_items_blinded.csv" `
  --answer_key_csv ".\evaluations\human_review\ddpm_v2_answer_key.csv" `
  --max_examples 30 `
  --seed 42 `
  --panel_size 256 `
  --title "ArchiveDiffusion DDPM v2 blinded review"

\## 21. evaluating the human review
python ".\scripts\human_evaluation_summary.py" `
  --ratings_csv ".\evaluations\human_review\ddpm2_human_review_ratings.csv" `
  --answer_key_csv ".\evaluations\human_review\ddpm_v2_answer_key.csv" `
  --output_dir ".\evaluations\human_review" `
  --output_prefix "ddpm2_human_review"

\## 22. Longer training with train / val splits
python ".\scripts\train_conditional_ddpm.py" `
  --dataset_dir ".\data\processed\synthetic_restoration\nosferatu_v0" `
  --metadata_file "metadata_with_splits.csv" `
  --train_split train `
  --val_split val `
  --output_dir ".\outputs\models\conditional_ddpm_v3_splitaware_optimised" `
  --image_size 128 `
  --batch_size 2 `
  --epochs 250 `
  --learning_rate 7.5e-5 `
  --num_train_timesteps 1000 `
  --seed 42 `
  --save_every_epochs 50 `
  --use_mlflow `
  --mlflow_tracking_uri "sqlite:///mlflow.db" `
  --mlflow_experiment_name "ArchiveDiffusion" `
  --run_name "conditional_ddpm_v3_splitaware_optimised_250epochs"

\## 23. train residual DDPM model
python ".\scripts\train_residual_conditional_ddpm.py" `
  --dataset_dir ".\data\processed\synthetic_restoration\nosferatu_v0" `
  --metadata_file "metadata_with_splits.csv" `
  --train_split train `
  --val_split val `
  --output_dir ".\outputs\models\residual_ddpm_v1_splitaware" `
  --image_size 128 `
  --batch_size 2 `
  --epochs 250 `
  --learning_rate 7.5e-5 `
  --num_train_timesteps 1000 `
  --seed 42 `
  --save_every_epochs 50 `
  --use_mlflow `
  --mlflow_tracking_uri "sqlite:///mlflow.db" `
  --mlflow_experiment_name "ArchiveDiffusion" `
  --run_name "residual_ddpm_v1_splitaware_250epochs"

\## 24. sample longer train test split
python ".\scripts\sample_conditional_ddpm.py" `
  --model_dir ".\outputs\models\conditional_ddpm_v3_splitaware_optimised\final" `
  --dataset_dir ".\data\processed\synthetic_restoration\nosferatu_v0" `
  --metadata_file "metadata_with_splits.csv" `
  --split test `
  --output_dir ".\outputs\predictions\conditional_ddpm_v3_splitaware_50steps" `
  --method_name "conditional_ddpm_v3_splitaware_50_steps" `
  --num_inference_steps 50

\## 25. sample residual DDPM model (full strength)
python ".\scripts\sample_residual_conditional_ddpm.py" `
  --model_dir ".\outputs\models\residual_ddpm_v1_splitaware\final" `
  --dataset_dir ".\data\processed\synthetic_restoration\nosferatu_v0" `
  --metadata_file "metadata_with_splits.csv" `
  --split test `
  --output_dir ".\outputs\predictions\residual_ddpm_v1_full_test_50steps" `
  --method_name "residual_ddpm_v1_full_test_50_steps" `
  --num_inference_steps 50 `
  --correction_strength 1.0

\## 26. sample residual DDPM model (conservative)
python ".\scripts\sample_residual_conditional_ddpm.py" `
  --model_dir ".\outputs\models\residual_ddpm_v1_splitaware\final" `
  --dataset_dir ".\data\processed\synthetic_restoration\nosferatu_v0" `
  --metadata_file "metadata_with_splits.csv" `
  --split test `
  --output_dir ".\outputs\predictions\residual_ddpm_v1_conservative_test_50steps" `
  --method_name "residual_ddpm_v1_conservative_test_50_steps" `
  --num_inference_steps 50 `
  --correction_strength 0.5

\## 27. Evaluate optimised DDPM
python ".\scripts\evaluate_restoration_outputs.py" `
  --prediction_manifest ".\outputs\predictions\conditional_ddpm_v3_splitaware_50steps\prediction_manifest.csv" `
  --output_metrics ".\evaluations\results\conditional_ddpm_v3_splitaware_50steps_metrics.csv" `
  --output_summary ".\evaluations\results\conditional_ddpm_v3_splitaware_50steps_summary.csv" `
  --output_grid ".\outputs\evaluation_grids\conditional_ddpm_v3_splitaware_50steps_examples.png"

\## 28. Evaluate full residual DDPM
python ".\scripts\evaluate_restoration_outputs.py" `
  --prediction_manifest ".\outputs\predictions\residual_ddpm_v1_full_test_50steps\prediction_manifest.csv" `
  --output_metrics ".\evaluations\results\residual_ddpm_v1_full_test_50steps_metrics.csv" `
  --output_summary ".\evaluations\results\residual_ddpm_v1_full_test_50steps_summary.csv" `
  --output_grid ".\outputs\evaluation_grids\residual_ddpm_v1_full_test_50steps_examples.png"

\## 29. Evaluate conservative residual DDPM
python ".\scripts\evaluate_restoration_outputs.py" `
  --prediction_manifest ".\outputs\predictions\residual_ddpm_v1_conservative_test_50steps\prediction_manifest.csv" `
  --output_metrics ".\evaluations\results\residual_ddpm_v1_conservative_test_50steps_metrics.csv" `
  --output_summary ".\evaluations\results\residual_ddpm_v1_conservative_test_50steps_summary.csv" `
  --output_grid ".\outputs\evaluation_grids\residual_ddpm_v1_conservative_test_50steps_examples.png"

\## 30. Generate evaluation sheets for residual DDPMs
python ".\scripts\create_human_review_sheet.py" `
  --metadata_file ".\data\processed\synthetic_restoration\nosferatu_v0\metadata_with_splits.csv" `
  --split test `
  --prediction_manifest ".\outputs\predictions\conditional_ddpm_nosferatu_v1_50steps\prediction_manifest.csv" `
  --prediction_manifest ".\outputs\predictions\conditional_ddpm_nosferatu_v1_100steps\prediction_manifest.csv" `
  --prediction_manifest ".\outputs\predictions\conditional_ddpm_v3_splitaware_50steps\prediction_manifest.csv" `
  --prediction_manifest ".\outputs\predictions\residual_ddpm_v1_full_test_50steps\prediction_manifest.csv" `
  --prediction_manifest ".\outputs\predictions\residual_ddpm_v1_conservative_test_50steps\prediction_manifest.csv" `
  --output_dir ".\evaluations\human_review\review_sheets\model_calibration_v1_blinded" `
  --output_csv ".\evaluations\human_review\model_calibration_v1_review_items_blinded.csv" `
  --answer_key_csv ".\evaluations\human_review\model_calibration_v1_answer_key.csv" `
  --max_examples 21 `
  --seed 123 `
  --panel_size 256 `
  --title "ArchiveDiffusion model calibration v1 blinded review"

\## 31. Evaluation summary for residual DDPM
python ".\scripts\human_evaluation_summary.py" `
  --ratings_csv ".\evaluations\human_review\model_calibration_v1_human_review_ratings.csv" `
  --answer_key_csv ".\evaluations\human_review\model_calibration_v1_answer_key.csv" `
  --output_dir ".\evaluations\human_review" `
  --output_prefix "model_calibration_v1"

\## 32. Loss curves for updated models
python ".\scripts\plot_training_curves.py" `
  --run_dir ".\outputs\models\residual_ddpm_v1_splitaware" `
  --output_path ".\outputs\training_curves\residual_ddpm_v1_splitaware_train_val_loss.png" `
  --title "Residual DDPM v1 split-aware train/validation loss"

python ".\scripts\plot_training_curves.py" `
  --run_dir ".\outputs\models\conditional_ddpm_v3_splitaware_optimised" `
  --output_path ".\outputs\training_curves\conditional_ddpm_v3_splitaware_optimised_train_val_loss.png" `
  --title "Conditional DDPM v3 split-aware optimised train/validation loss"