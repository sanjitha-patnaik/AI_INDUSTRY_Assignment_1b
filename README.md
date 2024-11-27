# AI_INDUSTRY_Assignment_1b
A repository for evaluating generative video models using Frechet Inception Distance for Videos (FID-Vid) and Frechet Video Distance (FVD) metrics.


# Video Evaluation Framework

This repository contains the implementation of an evaluation framework for synthetic human action recognition. The framework computes **FIDVID (Frechet Inception Distance for Videos)** and **FVD (Frechet Video Distance)** scores to evaluate the quality of synthetic video data.

---

## **Setup Instructions**

### 1. Clone the Repository
Clone the repository to your local system:
```bash
git clone <repository-link>
cd <repository-folder>
```

### 2. Install Dependencies
Install the required Python libraries using:
```bash
pip install -r requirements.txt
```

### 3. Run the Framework
Launch the Gradio interface by running:
```bash
python scores.py
```
A local server will start, and the URL (e.g., `http://127.0.0.1:7860`) will be displayed.

---

## **How to Use**

1. **Input Folders**
   - Enter the **absolute paths** for two folders containing the videos you want to compare:
     - **Folder 1**: Can contain either real or synthetic videos.
     - **Folder 2**: Can contain synthetic videos generated by a different method.

2. **Start Evaluation**
   - Click **Submit**. The framework will:
     - Load frames from corresponding videos in both folders.
     - Compute **FIDVID** and **FVD** scores for each pair of videos.

3. **View Results**
   - The results will be displayed in a table format with the following columns:
     - **Real Video**: Name of the video from Folder 1.
     - **Synthetic Video**: Name of the video from Folder 2.
     - **FIDVID Score**: Frame-level feature similarity score.
     - **FVD Score**: Spatiotemporal consistency score.

---

## **Expected Outputs**

For the input folders, the output table will look like this:

| **Real Video**      | **Synthetic Video**  | **FIDVID Score**      | **FVD Score**          |
|---------------------|----------------------|-----------------------|------------------------|
| Co_S1K1_fC6.avi     | Co_S8K1_fC8.avi     | 214.90584949027       | 1.935691333872461e-9   |
| Co_S1K1_fC7.avi     | Co_S8K2_fC17.avi    | 414.71024453973234    | 4.3130710279307615e-9  |
| Co_S1K1_fC8.avi     | Co_S8K2_fC18.avi    | 272.15644039741505    | 1.431334933393025e-8   |
| Co_S1K1_m3.avi      | Co_S8K2_fC19.avi    | 338.47200487184614    | 7.185410256372668e-9   |
| Co_S1K2_fC17.avi    | Co_S8K2_fC20.avi    | 337.47243716245356    | 3.581631395395524e-9   |

---

## **Understanding the Scores**

- **FIDVID (Frechet Inception Distance for Videos)**:
  - Measures frame-level similarity between two videos.
  - **Lower scores** indicate closer similarity between videos.

- **FVD (Frechet Video Distance)**:
  - Measures spatiotemporal consistency in video dynamics.
  - **Lower scores** indicate better temporal and motion similarity.

---

## **Best Practices**

1. **Ensure Correspondence**:
   - The videos in Folder 1 and Folder 2 should correspond (e.g., same action, same video ID).

2. **Input Quality**:
   - Videos should have consistent resolution and frame rates for accurate comparison.

3. **Interpreting Scores**:
   - A **low FIDVID and FVD score** means the synthetic video is close to the real or ideal baseline.

---


