## Dataset preparations: 
Download each dataset from the links below and place them in the datasets/ folder. The CPS dataset [1] is already included, and Spambase [2] will be fetched automatically at runtime.

1. Link for the metabric dataset [3, 4]:
https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric

2. Link for the CKD dataset [5]:
https://www.kaggle.com/dsv/8658224

3. Link for the CTGS dataset [6]:
https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175

## Run

Once the datasets are installed, you can train and evaluate our AFA method on the paper’s tabular datasets (Spambase, Metabric, CPS, CTGS, CKD).

1. Open `train_and_test_image.sh` and set the `dataset` parameter to your target dataset.  
   Example inside the file:
   ```bash
   # Options: spam, metabric, cps, ctgs, ckd
   dataset=cps
2.
    ```bash
    sh train_and_test_tabular.sh
    ```

## References
[1] Dickson, E., Grambsch, P., Fleming, T., Fisher, L., and Langworthy, A. Cirrhosis Patient Survival Prediction. UCI Machine Learning Repository, 1989. DOI: https://doi.org/10.24432/C5R02G.

[2] Hopkins, M., Reeber, E., Forman, G., , and Suermondt, J. Spambase. UCI Machine Learning Repository, 1999. DOI: https://doi.org/10.24432/C53G6X.

[3] Curtis, C., Shah, S. P., Chin, S.-F., Turashvili, G., Rueda, O. M., Dunning, M. J., Speed, D., Lynch, A. G., Samarajiwa, S., Yuan, Y., Graf, S., Ha, G., Haffari, G., ¨Bashashati, A., Russell, R., McKinney, S., Langerød,A., Green, A., Provenzano, E., Wishart, G., Pinder, S., Watson, P., Markowetz, F., Murphy, L., Ellis, I., Purushotham, A., Børresen-Dale, A.-L., Brenton, J. D., Tavare, S., Caldas, C., and Aparicio, S. The genomic and transcriptomic architecture of 2, 000 breast tumours reveals novel subgroups. Nature, 486(7403):346–352, April 2012. ISSN 1476-4687. doi: 10.1038/nature10983. URL http://dx.doi.org/10.1038/nature10983.

[4] Pereira, B., Chin, S.-F., Rueda, O. M., Vollan, H.-K. M., Provenzano, E., Bardwell, H. A., Pugh, M., Jones, L., Russell, R., Sammut, S.-J., Tsui, D. W. Y., Liu, B., Dawson, S.-J., Abraham, J., Northen, H., Peden, J. F., Mukherjee, A., Turashvili, G., Green, A. R., McKinney, S., Oloumi, A., Shah, S., Rosenfeld, N., Murphy, L., Bentley, D. R., Ellis, I. O., Purushotham, A., Pinder, S. E.,Børresen-Dale, A.-L., Earl, H. M., Pharoah, P. D., Ross, M. T., Aparicio, S., and Caldas, C. Erratum: The somatic mutation profiles of 2,433 breast cancers refine their genomic and transcriptomic landscapes. Nat. Commun., 7 (1):11908, June 2016.

[5] Kharoua, R. E. Chronic kidney disease dataset, 2024. URL https://www.kaggle.com/dsv/8658224.

[6] Hammer, S. M., Katzenstein, D. A., Hughes, M. D., Gundacker, H., Schooley, R. T., Haubrich, R. H., Henry, W. K., Lederman, M. M., Phair, J. P., Niu, M., Hirsch, M. S., and Merigan, T. C. A trial comparing nucleoside monotherapy with combination therapy in hiv-infected adults with cd4 cell counts from 200 to 500 per cubic millimeter. aids clinical trials group study 175 study team. The New England journal of medicine, 335 15:1081–90, 1996. URL https://api.semanticscholar.org/CorpusID:40754467.
