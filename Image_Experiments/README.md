## Link for FastSHAP models:
[Google Drive](https://drive.google.com/drive/folders/1ILhnf0PWkFpe258uUqy0BHSGnbdHE9SD?usp=sharing)

Install the FastSHAP [1] checkpoints with gdown:

gdown --folder "https://drive.google.com/drive/folders/1ILhnf0PWkFpe258uUqy0BHSGnbdHE9SD" -O ckpts/

## Run


Once the FastSHAP checkpoints are installed, you can train and evaluate our AFA method on the paper’s image datasets (CIFAR-10 [2], CIFAR-100 [2], BloodMNIST [3, 4, 5], ImageNette [6]).

1. Open `train_and_test_image.sh` and set the `dataset` parameter to your target dataset.  
   Example inside the file:
   ```bash
   # Options: cifar10, cifar100, bloodmnist, imagenette
   dataset=cifar10
2.
    ```bash
    sh train_and_test_image.sh
    ```

## References
[1] Jethani, N., Sudarshan, M., Covert, I. C., Lee, S.-I., and Ranganath, R. FastSHAP: Real-time shapley value estimation. In International Conference on Learning Representations, 2022.

[2] Krizhevsky, A. Learning multiple layers of features from tiny images. pp. 32–33, 2009. URL https://www.cs.toronto.edu/˜kriz/ learning-features-2009-TR.pdf.

[3] Acevedo, A., Merino, A., Alférez, S., Molina, Á., Boldú, L., & Rodellar, J. (2020). A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. Data in brief, 30, 105474.

[4] Yang, J., Shi, R., and Ni, B. Medmnist classification decathlon: A lightweight automl benchmark for medical image analysis. In IEEE 18th International Symposium on Biomedical Imaging (ISBI), pp. 191–195, 2021.

[5] Yang, J., Shi, R., and Ni, B. Medmnist classification decathlon: A lightweight automl benchmark for medicalimage analysis. In IEEE 18th International Symposium on Biomedical Imaging (ISBI), pp. 191–195, 2021.

[6] Howard, J. Imagenette: A smaller subset of 10 easily classified classes from imagenet, March 2019. URL https://github.com/fastai/imagenette.