# Vision and Cognitive Systems Final Project

## Utilizzo
E' richiesta la versione `3.6` di python ed è fortemente consigliato l'utilizzo di un virtual
 environment.
 
### Setup progetto e virtual environment
E' possibile utilizzare tools come `virtualenv`, `conda` o la sua versione light `miniconda`. Di
seguito le istruzioni per la creazione di un virtual environment con `miniconda` (che è possibile
installare tramite la [guida ufficiale](https://docs.conda.io/en/latest/miniconda.html#installing)).

1. Clonare il progetto da GitHub
    ```bash
       git clone git@github.com:enrico-ghidoni/vcs-final-project.git
       cd vcs-final-project
    ```
2. Creare un ambiente dedicato al progetto con una versione di python >=3.6
    ```bash
       conda create -n vcs-final-project python=3.6
    ```
3. Attivare l'ambiente appena creato
    ```bash
       conda activate vcs-final-project
    ```
4. Installare il progetto localmente con pip
    ```bash
       pip install -e .
    ```

E' possibile specificare un nome diverso da `vcs-final-project` al passo 2 per l'ambiente
 virtuale. Contrariamente a `virtualenv`, `conda` non crea una directory dove specificato ma all
 'interno della propria installazione.