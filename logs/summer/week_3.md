# Week 3
## Summer

I implemented CSP + LDA classification pipeline for left vs right hand motor imagery using raw EEG motor imagery data. I used subject 1 from the PhysioNet movement dataset through MNE. Through the pipeline, was able to get an average accuracy of 0.73, with scores ranging from 0.583 to 0.917. Using only the imagined motor movement data (runs 7 and 8), the accuracy drops to 0.63. Using only the real motor movement data (runs 3 and 4), the accuracy decreases to 0.70.

### Files

- notebooks/week_3/csp_lda_competition.py