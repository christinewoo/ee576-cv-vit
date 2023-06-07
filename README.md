# Pokemon Classifier

### Dataset
Please download [Pokemon Dataset](https://www.kaggle.com/datasets/lantian773030/pokemonclassification)


### Environment
This repo is tested on our local environment (python=3.11, cuda=11.7, pytorch=2.0.1), and we recommend you to use Anaconda to create a vitural environment:

1. Create a conda environment:
    ```
    conda create -n pokemon_vit python=3.11
    conda activate pokemon_vit
    ```

2. Install [Pytorch](https://pytorch.org/get-started/locally/) and torchvision according to your CUDA version

    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    ```

All credits for dataset and libraries goes to the owners listed in references.


### Training, Inference, GUI
    conda activate pokemon_vit
    cd models

1. Train ViT implemented from scratch
    ```
    python vit.py
    ```

2. Fine-Tune Pokemon Classifier on Pokemon dataset
    ```
    python vit_pokemon.py
    ```

3. Inference Pokemon Classifier (code)
    ```
    python vit_pokemon_inf.py
    ```

4. Inference Pokemon Classifier (GUI)
    ```
    python gui.py
    ```


### References
https://github.com/imjeffhi4/pokemon-classifier
https://www.kaggle.com/datasets/lantian773030/pokemonclassification
https://github.com/huggingface/transformers/tree/main/src/transformers/models/vit
https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c