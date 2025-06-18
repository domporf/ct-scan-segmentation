# ct-scan-segmentation

Python-based pipeline for semi-automated segmentation of true and false lumen in Type B Aortic Dissection (TBAD) CT angiography scans.

This project integrates the Segment Anything Model (SAM) into a semi-automated pipeline for segmenting the true and false lumen in CT angiography scans of patients with Type B Aortic Dissection (TBAD).

---

## Project Overview

- **Clinical Problem:** TBAD is a life-threatening condition requiring accurate measurement of the true and false lumen to guide treatment.  
- **Goal:** Reduce manual segmentation workload and improve consistency in lumen volume assessment using a zero-shot deep learning model.  
- **Scope:** 40 patient scans segmented by a board-certified radiologist and compared to SAM-assisted segmentation.
- **Publication:** Findings were published by the European Society of Radioligy ([Publication](https://epos.myesr.org/poster/esr/ecr2024/C-14868))


---

## Key Features

- Python 3.10.10 pipeline with custom per-slice segmentation logic  
- Integration of Meta AI's Segment Anything Model (SAM)  
- Minimal user input (≤10 mouse clicks per scan)  
- Statistical validation: Dice scores, linear regression, Bland-Altman analysis, and two-sample t-tests  
- Auto-generated visualization of overlays and volume metrics

---

## Repository Structure

<pre> Repository Structure 
  ct-scan-segmentation/ 
  ├── sam_pipeline/ # Python scripts for SAM integration and segmentation 
  ├── results/ # Scripts for calculating segmentation accuracy 
  ├── poster/ # Research poster and publication link 
  └── README.md # Project overview and instructions  </pre>

---

## Methods

- **Segmentation Tool:** [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) applied to CT slices  
- **Validation:** Expert-labeled volumes obtained using TeraRecon Intuition  
- **Analysis:** Dice similarity coefficient, regression, and statistical comparison of volumes  

---

## Dependencies

- Python 3.10  
- PyTorch  
- OpenCV  
- NumPy, SciPy  
- Matplotlib  
- SAM (Meta AI) dependencies  

---

## Use Case

Designed to support radiologists, researchers, and engineers working in cardiovascular imaging and AI-assisted diagnostics.

---

## Sample Output

Volume comparison charts and statistical outputs are included in the `results/` folder.

---

## Note on Data Privacy

Due to HIPAA regulations, CT angiography scan data is not included in this repository.

---

## Author

**Dominic Millan Profit***  
Environmental Engineer @ Tufts University  
Contact: dprofit67@gmail.com  

Project completed during the Stanford Institutes of Medicine Summer Research Program (SIMR) with Bethzaida Sandoval Valle*, Ashish Manohar, PhD, Gabriel Mistelbauer, PhD, Dominik Fleischmann, MD, and Koen Nieman, MD.  
\*Contributed equally.
